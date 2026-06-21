import os
import time as pytime
import logging
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel

logging.basicConfig(
    level=logging.INFO, 
    format='[*] %(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load .env dari direktori tempat script berada
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# ClickHouse HTTP Settings (Menggunakan SPARK_CH_USER dan SPARK_CH_PASSWORD)
CH_HOST = os.getenv('SPARK_CH_HOST', 'clickhouse3')
CH_PORT = os.getenv('SPARK_CH_PORT', '8123')
CH_USER = os.getenv('SPARK_CH_USER', 'spark_user')
CH_PASSWORD = os.getenv('SPARK_CH_PASSWORD', 'spark_secure_password_123!')
CH_DB = os.getenv('CLICKHOUSE_DB', 'bigdata')

# Jika berjalan di dalam Docker container (seperti Spark master), hubungkan langsung ke kontainer ClickHouse
if os.path.exists('/.dockerenv') or os.getenv('SPARK_HOME') is not None:
    CH_HOST = 'clickhouse3'
    CH_PORT = '8123'

def execute_clickhouse_ddl(spark, jdbc_url, user, password, query):
    try:
        sc = spark.sparkContext
        sc._jvm.java.lang.Class.forName("com.clickhouse.jdbc.ClickHouseDriver")
        properties = sc._jvm.java.util.Properties()
        properties.setProperty("user", user)
        properties.setProperty("password", password)
        conn = sc._jvm.java.sql.DriverManager.getConnection(jdbc_url, properties)
        stmt = conn.createStatement()
        stmt.execute(query)
        stmt.close()
        conn.close()
        return True
    except Exception as e:
        logging.warning(f"Gagal menjalankan query DDL/JDBC: {e}")
        return False

def setup_predictions_table(spark, jdbc_url, user, password):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions (
        time DateTime,
        close_prediction Nullable(Float64)
    ) ENGINE = ReplacingMergeTree()
    PARTITION BY toYYYYMM(time)
    ORDER BY (time);
    """
    if execute_clickhouse_ddl(spark, jdbc_url, user, password, create_table_query):
        logging.info("Tabel 'bitcoin_predictions' siap digunakan (ReplacingMergeTree).")
    else:
        logging.warning("Tidak dapat membuat/memverifikasi tabel (mungkin hak akses DDL terbatas).")

def main():
    # 1. Inisialisasi Spark Session
    if os.path.exists('/.dockerenv') or os.getenv('SPARK_HOME') is not None:
        jdbc_url = f"jdbc:clickhouse://clickhouse3:8123/{CH_DB}"
    else:
        jdbc_url = f"jdbc:clickhouse://localhost:8125/{CH_DB}"

    spark = SparkSession.builder \
        .appName("Bitcoin_Batch_Prediction") \
        .config("spark.master", "spark://spark-master3:7077") \
        .getOrCreate() 
    spark.sparkContext.setLogLevel("WARN")

    # 2. Inisialisasi Tabel Prediksi via Spark JDBC
    setup_predictions_table(spark, jdbc_url, CH_USER, CH_PASSWORD)

    # 3. Dapatkan waktu terakhir yang sudah diprediksi untuk idempotensi & inkremental
    latest_time_res = "1970-01-01 00:00:00"
    try:
        df_max = spark.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("user", CH_USER) \
            .option("password", CH_PASSWORD) \
            .option("dbtable", "(SELECT max(time) as max_time FROM bigdata.bitcoin_predictions) AS max_time_query") \
            .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
            .load()
        
        rows = df_max.collect()
        if rows and rows[0]['max_time']:
            res = rows[0]['max_time']
            if hasattr(res, 'strftime'):
                latest_time_res = res.strftime('%Y-%m-%d %H:%M:%S')
            else:
                latest_time_res = str(res)
    except Exception as e:
        logging.info(f"Tabel bitcoin_predictions kosong atau belum siap: {e}")

    logging.info(f"Mengambil data baru dengan time > '{latest_time_res}'")

    # 4. Ambil data baru secara native menggunakan Spark JDBC
    query = f"""
    (
        SELECT 
            time, close, volume, volume_lag_1, close_delta,
            dist_to_mean_5, dist_to_mean_60, dist_to_max_60, dist_to_min_60,
            close_std_60, volume_sum_60, close_delta_60
        FROM bigdata.bitcoin_features
        WHERE time > toDateTime('{latest_time_res}')
    ) AS features_query
    """
    
    logging.info("Mengambil fitur baru dari ClickHouse via Spark JDBC...")
    spark_df = spark.read \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", query) \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .load()
    
    # Cek apakah data kosong
    if spark_df.rdd.isEmpty():
        logging.info("Tidak ada data baru untuk diprediksi. Keluar...")
        spark.stop()
        return

    # Bersihkan data: Hapus baris dengan nilai null atau NaN
    spark_df = spark_df.dropna()

    # Cek lagi setelah pembersihan
    if spark_df.rdd.isEmpty():
        logging.info("Tidak ada data setelah pembersihan (semua null/NaN). Keluar...")
        spark.stop()
        return

    logging.info("Memulai Spark untuk prediksi...")

    # 5. Load Model dari HDFS
    hdfs_path = "hdfs://namenode3:9000/models/random_forest_reg"
    logging.info(f"Memuat model dari HDFS: {hdfs_path}")
    model = PipelineModel.load(hdfs_path)

    # 6. Siapkan data untuk model (tambahkan time_unix)
    spark_df = spark_df.withColumn("time_unix", col("time").cast("long"))

    # 7. Jalankan Inference Model
    predictions = model.transform(spark_df)

    # 8. Hitung close_prediction = close + prediction, dan geser time ke +60 detik
    result_spark_df = predictions.withColumn("close_prediction", col("close") + col("prediction")) \
                                 .withColumn("time", (col("time_unix") + 60).cast("timestamp")) \
                                 .select("time", "close_prediction")

    # 9. Tulis ke ClickHouse secara native via Spark JDBC
    logging.info("Menulis prediksi kembali ke ClickHouse via Spark JDBC...")
    result_spark_df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "bigdata.bitcoin_predictions") \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()

    # 10. Jalankan OPTIMIZE untuk deduplikasi ReplacingMergeTree jika diizinkan
    try:
        logging.info("Deduplikasi final (OPTIMIZE TABLE)...")
        execute_clickhouse_ddl(spark, jdbc_url, CH_USER, CH_PASSWORD, "OPTIMIZE TABLE bigdata.bitcoin_predictions FINAL")
        logging.info("Deduplikasi selesai.")
    except Exception as e:
        logging.warning(f"OPTIMIZE dilewati (mungkin hak akses ALTER terbatas): {e}")

    logging.info("Proses batch prediction selesai dengan sukses!")
    spark.stop()

if __name__ == '__main__':
    main()
