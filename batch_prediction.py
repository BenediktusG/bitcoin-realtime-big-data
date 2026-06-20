import os
import time as pytime
import logging
from dotenv import load_dotenv
import clickhouse_connect
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
CH_HOST = os.getenv('CH_HOST', 'localhost')
CH_PORT = os.getenv('CH_PORT', '8125')
CH_USER = os.getenv('SPARK_CH_USER', 'spark_user')
CH_PASSWORD = os.getenv('SPARK_CH_PASSWORD', 'spark_secure_password_123!')
CH_DB = os.getenv('CLICKHOUSE_DB', 'bigdata')

# Jika berjalan di dalam Docker container (seperti Spark master), hubungkan langsung ke kontainer ClickHouse
if os.path.exists('/.dockerenv') or os.getenv('SPARK_HOME') is not None:
    CH_HOST = 'clickhouse3'
    CH_PORT = '8123'

def get_clickhouse_client():
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD, database=CH_DB
        )
        return client
    except Exception as e:
        logging.error(f"Gagal terhubung ke ClickHouse: {e}")
        raise

def setup_predictions_table(client):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions (
        time DateTime,
        close Float64,
        prediction Float64,
        predicted_close Float64
    ) ENGINE = ReplacingMergeTree()
    PARTITION BY toYYYYMM(time)
    ORDER BY (time);
    """
    try:
        client.command(create_table_query)
        logging.info("Tabel 'bitcoin_predictions' siap digunakan (ReplacingMergeTree).")
    except Exception as e:
        logging.warning(f"Tidak dapat membuat/memverifikasi tabel (mungkin hak akses DDL terbatas): {e}")

def main():
    # 1. Inisialisasi ClickHouse Client
    ch_client = get_clickhouse_client()
    setup_predictions_table(ch_client)

    # Dapatkan waktu terakhir yang sudah diprediksi untuk idempotensi & inkremental
    latest_time_res = "1970-01-01 00:00:00"
    try:
        res = ch_client.command("SELECT max(time) FROM bigdata.bitcoin_predictions")
        if res and res != 0:
            if hasattr(res, 'strftime'):
                latest_time_res = res.strftime('%Y-%m-%d %H:%M:%S')
            else:
                latest_time_res = str(res)
    except Exception as e:
        logging.info(f"Tabel bitcoin_predictions kosong atau belum siap: {e}")

    logging.info(f"Mengambil data baru dengan time > '{latest_time_res}'")

    # 2. Ambil data baru sebagai Pandas DataFrame menggunakan clickhouse-connect
    query = f"""
    SELECT 
        time, close, volume, volume_lag_1, close_delta,
        dist_to_mean_5, dist_to_mean_60, dist_to_max_60, dist_to_min_60,
        close_std_60, volume_sum_60, close_delta_60
    FROM bigdata.bitcoin_features
    WHERE time > toDateTime('{latest_time_res}')
    ORDER BY time ASC
    """
    
    logging.info("Mengambil fitur baru dari ClickHouse via clickhouse-connect...")
    features_pandas_df = ch_client.query_df(query)
    
    row_count = len(features_pandas_df)
    if row_count == 0:
        logging.info("Tidak ada data baru untuk diprediksi. Keluar...")
        ch_client.close()
        return

    logging.info(f"Ditemukan {row_count} data baru. Memulai Spark untuk prediksi...")

    # 3. Inisialisasi Spark Session
    spark = SparkSession.builder \
        .appName("Bitcoin_Batch_Prediction") \
        .config("spark.master", "spark://spark-master3:7077") \
        .getOrCreate() 
    spark.sparkContext.setLogLevel("WARN")

    # 4. Load Model dari HDFS
    hdfs_path = "hdfs://namenode3:9000/models/rf_regressor_optim_model"
    logging.info(f"Memuat model dari HDFS: {hdfs_path}")
    model = PipelineModel.load(hdfs_path)

    # 5. Konversi Pandas DataFrame ke Spark DataFrame
    # time_unix diperlukan untuk komputasi window atau internal model jika ada, buat kolomnya
    spark_df = spark.createDataFrame(features_pandas_df)
    # Tambahkan time_unix (untuk time series split / indexing jika diperlukan oleh model)
    spark_df = spark_df.withColumn("time_unix", col("time").cast("long"))

    # 6. Jalankan Inference Model
    predictions = model.transform(spark_df)

    # 7. Hitung predicted_close = close + prediction
    result_spark_df = predictions.withColumn("predicted_close", col("close") + col("prediction")) \
                                 .select("time", "close", "prediction", "predicted_close")

    # 8. Konversi hasil kembali ke Pandas DataFrame untuk ditulis via clickhouse-connect
    logging.info("Mengonversi hasil prediksi kembali ke Pandas...")
    result_pandas_df = result_spark_df.toPandas()

    # 9. Tulis ke ClickHouse
    logging.info("Menulis prediksi kembali ke ClickHouse via clickhouse-connect...")
    ch_client.insert_df("bigdata.bitcoin_predictions", result_pandas_df)

    # 10. Jalankan OPTIMIZE untuk deduplikasi ReplacingMergeTree jika diizinkan
    try:
        logging.info("Deduplikasi final (OPTIMIZE TABLE)...")
        ch_client.command("OPTIMIZE TABLE bigdata.bitcoin_predictions FINAL")
        logging.info("Deduplikasi selesai.")
    except Exception as e:
        logging.warning(f"OPTIMIZE dilewati (mungkin hak akses ALTER terbatas): {e}")

    ch_client.close()
    logging.info("Proses batch prediction selesai dengan sukses!")
    spark.stop()

if __name__ == '__main__':
    main()
