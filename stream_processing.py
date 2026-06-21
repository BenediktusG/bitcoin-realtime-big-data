import os
import logging
import datetime
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, unix_timestamp, window, expr, to_timestamp, lead, lag, mean as _mean, max as _max, min as _min, stddev as _stddev, sum as _sum
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType
from pyspark.ml import PipelineModel
from pyspark.sql.window import Window

logging.basicConfig(
    level=logging.INFO, 
    format='[*] %(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

KAFKA_URL = os.getenv('SPARK_KAFKA_URL', '127.0.0.1:9094')
TOPIC_NAME = 'bitcoin-orders'

SPARK_CH_HOST = os.getenv('SPARK_CH_HOST', 'localhost')
SPARK_CH_PORT = os.getenv('SPARK_CH_PORT', '8125')
CH_USER = os.getenv('SPARK_CH_USER', 'spark_user')
CH_PASSWORD = os.getenv('SPARK_CH_PASSWORD', 'spark_secure_password_123!')
CH_DB = os.getenv('CLICKHOUSE_DB', 'bigdata')

if os.path.exists('/.dockerenv') or os.getenv('SPARK_HOME') is not None:
    SPARK_CH_HOST = 'clickhouse3'
    SPARK_CH_PORT = '8123'
    KAFKA_URL = 'kafka3:29092'

jdbc_url = f"jdbc:clickhouse://{SPARK_CH_HOST}:{SPARK_CH_PORT}/{CH_DB}"

schema = StructType([
    StructField("time", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True)
])

def process_micro_batch(batch_df, batch_id, spark, model):
    if batch_df.isEmpty():
        return
        
    logging.info(f"Memproses Micro-Batch ID: {batch_id} (Jumlah baris: {batch_df.count()})")
    
    # Konversi time dari String ke Timestamp
    batch_df = batch_df.withColumn("time", to_timestamp("time", "yyyy-MM-dd HH:mm:ss"))
    
    # Simpan raw data ke bitcoin_orders
    batch_df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", "bigdata.bitcoin_orders") \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()
    logging.info("Tersimpan raw data ke bitcoin_orders.")

    # Untuk feature engineering (butuh 60 baris sebelumnya), kita akan ambil 65 baris terakhir dari bitcoin_orders
    history_query = f"""
    (SELECT time, close, volume FROM bigdata.bitcoin_orders ORDER BY time DESC LIMIT 65) AS hist
    """
    history_df = spark.read \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", history_query) \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .load()
    
    # Hitung fitur
    window_past_60 = Window.orderBy("time").rowsBetween(-60, 0)
    window_past_5 = Window.orderBy("time").rowsBetween(-5, 0)
    window_exact = Window.orderBy("time")
    
    feat_df = history_df.withColumn("close_mean_60", _mean("close").over(window_past_60)) \
                        .withColumn("close_max_60", _max("close").over(window_past_60)) \
                        .withColumn("close_min_60", _min("close").over(window_past_60)) \
                        .withColumn("close_std_60", _stddev("close").over(window_past_60)) \
                        .withColumn("volume_sum_60", _sum("volume").over(window_past_60)) \
                        .withColumn("close_mean_5", _mean("close").over(window_past_5)) \
                        .withColumn("close_lag_60", lag("close", 60).over(window_exact)) \
                        .withColumn("volume_lag_1", lag("volume", 1).over(window_exact)) \
                        .withColumn("close_lag_1", lag("close", 1).over(window_exact))
                        
    feat_df = feat_df.withColumn("close_delta", col("close") - col("close_lag_1")) \
                     .withColumn("close_delta_60", col("close") - col("close_lag_60")) \
                     .withColumn("dist_to_mean_60", (col("close") - col("close_mean_60")) / col("close_mean_60")) \
                     .withColumn("dist_to_max_60", (col("close_max_60") - col("close")) / col("close")) \
                     .withColumn("dist_to_min_60", (col("close") - col("close_min_60")) / col("close_min_60")) \
                     .withColumn("dist_to_mean_5", (col("close") - col("close_mean_5")) / col("close_mean_5"))

    # Ambil baris yang sesuai dengan data baru di micro-batch ini
    min_time = batch_df.select(_min("time")).collect()[0][0]
    new_features_df = feat_df.filter(col("time") >= min_time).dropna()
    
    if new_features_df.isEmpty():
        logging.info("Data history belum cukup untuk menghasilkan fitur 60-baris. Menunggu batch berikutnya...")
        return
        
    final_features_df = new_features_df.select(
        "time", "close", "volume", "volume_lag_1", "close_delta",
        "dist_to_mean_5", "dist_to_mean_60", "dist_to_max_60", "dist_to_min_60",
        "close_std_60", "volume_sum_60", "close_delta_60"
    )

    # 1. Simpan fitur ke ClickHouse
    final_features_df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", "bigdata.bitcoin_features") \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()
    logging.info("Tersimpan fitur yang dihitung ke bitcoin_features.")
    
    # 2. Prediksi model
    pred_input_df = final_features_df.withColumn("time_unix", unix_timestamp(col("time")))
    predictions = model.transform(pred_input_df)
    
    # 3. Hitung absolute close prediction (time + 60s)
    result_df = predictions.withColumn("close_prediction", col("close") + col("prediction")) \
                           .withColumn("time", (col("time_unix") + 60).cast(TimestampType())) \
                           .select("time", "close_prediction")
                           
    result_df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", "bigdata.bitcoin_predictions") \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()
        
    logging.info("Prediksi Spark berhasil disimpan ke bitcoin_predictions.")

def main():
    logging.info("Inisialisasi Spark Session untuk Structured Streaming...")
    spark = SparkSession.builder \
        .appName("Bitcoin_Structured_Streaming") \
        .config("spark.master", "spark://spark-master3:7077") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark_streaming_checkpoints") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    hdfs_path = "hdfs://namenode3:9000/models/random_forest_reg"
    logging.info(f"Memuat model dari HDFS: {hdfs_path}")
    model = PipelineModel.load(hdfs_path)

    logging.info(f"Membaca stream dari Kafka {KAFKA_URL} topik {TOPIC_NAME}...")
    # Read from Kafka using Spark Structured Streaming
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_URL) \
        .option("subscribe", TOPIC_NAME) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON value
    parsed_df = kafka_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json("value", schema).alias("data")) \
        .select("data.*")

    # Start stream and process using foreachBatch
    query = parsed_df.writeStream \
        .foreachBatch(lambda df, epoch_id: process_micro_batch(df, epoch_id, spark, model)) \
        .start()

    logging.info("Structured Streaming berjalan. Tekan Ctrl+C untuk berhenti.")
    query.awaitTermination()

if __name__ == '__main__':
    main()

