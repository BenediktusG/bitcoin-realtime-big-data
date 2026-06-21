import os
import json
import logging
import datetime
from dotenv import load_dotenv

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, unix_timestamp, to_timestamp, lit
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType
from pyspark.ml import PipelineModel
from pyspark.sql.streaming.state import GroupStateTimeout, GroupState

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

out_schema = StructType([
    StructField("time", TimestampType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
    StructField("volume_lag_1", DoubleType(), True),
    StructField("close_delta", DoubleType(), True),
    StructField("dist_to_mean_5", DoubleType(), True),
    StructField("dist_to_mean_60", DoubleType(), True),
    StructField("dist_to_max_60", DoubleType(), True),
    StructField("dist_to_min_60", DoubleType(), True),
    StructField("close_std_60", DoubleType(), True),
    StructField("volume_sum_60", DoubleType(), True),
    StructField("close_delta_60", DoubleType(), True)
])

state_schema = StructType([
    StructField("history_json", StringType(), True)
])

def process_state(key, pdfs, state):
    if state.hasTimedOut:
        state.remove()
        return iter([])

    if state.exists:
        history = json.loads(state.get()[0])
    else:
        history = []
        
    out_records = []
    
    for pdf in pdfs:
        for idx, row in pdf.iterrows():
            record = {
                'time': str(row['time']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            history.append(record)
            if len(history) > 61:
                history.pop(0)
                
            if len(history) == 61:
                current = history[-1]
                lag_1 = history[-2]
                lag_60 = history[0]
                
                close_vals = [h['close'] for h in history[1:]]
                vol_vals = [h['volume'] for h in history[1:]]
                close_vals_5 = close_vals[-5:]
                
                mean_5 = sum(close_vals_5) / 5.0
                mean_60 = sum(close_vals) / 60.0
                max_60 = max(close_vals)
                min_60 = min(close_vals)
                
                import math
                variance = sum((x - mean_60) ** 2 for x in close_vals) / 60.0
                std_60 = math.sqrt(variance)
                sum_vol_60 = sum(vol_vals)
                
                features = {
                    'time': pd.Timestamp(current['time']),
                    'close': current['close'],
                    'volume': current['volume'],
                    'volume_lag_1': lag_1['volume'],
                    'close_delta': current['close'] - lag_1['close'],
                    'dist_to_mean_5': (current['close'] - mean_5) / mean_5,
                    'dist_to_mean_60': (current['close'] - mean_60) / mean_60,
                    'dist_to_max_60': (max_60 - current['close']) / current['close'],
                    'dist_to_min_60': (current['close'] - min_60) / min_60,
                    'close_std_60': std_60,
                    'volume_sum_60': sum_vol_60,
                    'close_delta_60': current['close'] - lag_60['close']
                }
                out_records.append(features)
                
    state.update((json.dumps(history),))
    
    if out_records:
        yield pd.DataFrame(out_records)
    else:
        yield pd.DataFrame(columns=[f.name for f in out_schema.fields])

def save_and_predict(batch_df, batch_id, model):
    if batch_df.isEmpty():
        return
        
    logging.info(f"Micro-Batch ID: {batch_id} - Menyimpan fitur dan memprediksi...")

    # 1. Simpan fitur ke ClickHouse
    batch_df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", "bigdata.bitcoin_features") \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()
    
    # 2. Prediksi model
    pred_input_df = batch_df.withColumn("time_unix", unix_timestamp(col("time")))
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
        
    logging.info("Prediksi Stateful Stream berhasil disimpan.")

def save_raw_orders(batch_df, batch_id):
    if batch_df.isEmpty(): return
    batch_df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("user", CH_USER) \
        .option("password", CH_PASSWORD) \
        .option("dbtable", "bigdata.bitcoin_orders") \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()

def main():
    logging.info("Inisialisasi Spark Session untuk Stateful Structured Streaming...")
    spark = SparkSession.builder \
        .appName("Bitcoin_Stateful_Streaming") \
        .config("spark.master", "spark://spark-master3:7077") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark_streaming_checkpoints") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    hdfs_path = "hdfs://namenode3:9000/models/random_forest_reg"
    logging.info(f"Memuat model dari HDFS: {hdfs_path}")
    model = PipelineModel.load(hdfs_path)

    logging.info(f"Membaca stream dari Kafka {KAFKA_URL} topik {TOPIC_NAME}...")
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_URL) \
        .option("subscribe", TOPIC_NAME) \
        .option("startingOffsets", "latest") \
        .load()

    parsed_df = kafka_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json("value", schema).alias("data")) \
        .select("data.*")
        
    parsed_df = parsed_df.withColumn("time", to_timestamp("time", "yyyy-MM-dd HH:mm:ss"))

    # Stream 1: Menyimpan data mentah ke bitcoin_orders
    query_raw = parsed_df.writeStream \
        .foreachBatch(save_raw_orders) \
        .start()

    # Stream 2: Proses stateful feature engineering & Prediksi
    stateful_features_df = parsed_df \
        .withColumn("routing_key", lit("bitcoin")) \
        .groupBy("routing_key") \
        .applyInPandasWithState(
            process_state,
            outputStructType=out_schema,
            stateStructType=state_schema,
            outputMode="append",
            timeoutConf="NoTimeout"
        )

    query_predict = stateful_features_df.writeStream \
        .foreachBatch(lambda df, epoch_id: save_and_predict(df, epoch_id, model)) \
        .start()

    logging.info("Stateful Structured Streaming berjalan. Tekan Ctrl+C untuk berhenti.")
    query_raw.awaitTermination()
    query_predict.awaitTermination()

if __name__ == '__main__':
    main()

