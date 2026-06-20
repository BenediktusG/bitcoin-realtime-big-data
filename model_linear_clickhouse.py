import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from dotenv import load_dotenv

load_dotenv()

print("[1] Memulai Spark Session (Linear Regression)...")
spark = SparkSession.builder \
    .appName("Bitcoin_LinReg_Forecast") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate() 
spark.sparkContext.setLogLevel("WARN")

print("[2] Mengambil Data & Feature Engineering dari ClickHouse...")
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(os.getenv("CLICKHOUSE_DB", "bigdata"))

query = """
(
    -- Membuang kolom yang tidak diperlukan, setara dengan df.drop(...)
    SELECT * EXCEPT(date_part, close_lag_60, next_close_60)
    FROM (
        SELECT 
            *,
            (close - close_lag_60) AS close_delta_60,
            (close - close_mean_60) / close_mean_60 AS dist_to_mean_60,
            (close_max_60 - close) / close AS dist_to_max_60,
            (close - close_min_60) / close_min_60 AS dist_to_min_60,
            (next_close_60 - close) AS target
        FROM (
            SELECT 
                *,
                toUnixTimestamp(time) AS time_unix,
                toDate(time) AS date_part,

                -- A. Fitur Masa Lalu
                avg(close) OVER w_past_60 AS close_mean_60,
                max(close) OVER w_past_60 AS close_max_60,
                min(close) OVER w_past_60 AS close_min_60,
                stddevSamp(close) OVER w_past_60 AS close_std_60,
                sum(volume) OVER w_past_60 AS volume_sum_60,
                lag(close, 60) OVER w_exact AS close_lag_60,

                -- B. TARGET REGRESI
                lead(close, 60) OVER w_exact AS next_close_60
                
            FROM bitcoin_features
            WHERE time >= toDateTime('2025-01-01 00:00:00')
            WINDOW 
                w_exact AS (PARTITION BY toDate(time) ORDER BY toUnixTimestamp(time)),
                w_past_60 AS (PARTITION BY toDate(time) ORDER BY toUnixTimestamp(time) RANGE BETWEEN 60 PRECEDING AND CURRENT ROW)
        )
        -- Setara dengan df.dropna()
        WHERE close_mean_60 IS NOT NULL 
          AND close_std_60 IS NOT NULL 
          AND close_lag_60 IS NOT NULL 
          AND next_close_60 IS NOT NULL
    )
) AS btc_data
"""

df = spark.read.format("jdbc").option("url", ch_url) \
    .option("user", os.getenv('CLICKHOUSE_USER', 'default')) \
    .option("password", os.getenv('CLICKHOUSE_PASSWORD', '')) \
    .option("dbtable", query).option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .option("partitionColumn", "time_unix").option("lowerBound", "1735689600") \
    .option("upperBound", "1775839739").option("numPartitions", "8").load()

print("[3] Membagi Data (Time-Series Split)...")
quantiles = df.approxQuantile("time_unix", [0.8], 0.01)
split_time_numeric = quantiles[0]

train_df = df.filter(col("time_unix") <= split_time_numeric)
test_df = df.filter(col("time_unix") > split_time_numeric)

print("[4] Membangun ML Pipeline Regresi...")
fitur_kolom = [
    # 'open', 'high', 'low', 'close', 'volume', 
    'volume', 
    'close_lag_1', 'volume_lag_1', 'close_roll_mean_5', 'close_delta',
    'dist_to_mean_60', 'dist_to_max_60', 'dist_to_min_60', 
    'close_std_60', 'volume_sum_60', 'close_delta_60'
]

assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="raw_features")

# PENTING: StandardScaler WAJIB untuk Linear Regression agar bobot (weight) adil
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

# Inisialisasi Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="target", maxIter=100, regParam=0.1, elasticNetParam=0.5)
pipeline = Pipeline(stages=[assembler, scaler, lr])

print("[5] Melatih Model Linear Regression...")
model = pipeline.fit(train_df)

print("[6] Evaluasi Model...")
predictions = model.transform(test_df)

# Menggunakan Evaluator khusus Regresi
evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction")

rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print("\n=== HASIL EVALUASI LINEAR REGRESSION ===")
print("RMSE (Root Mean Squared Error) : {:.4f}".format(rmse))
print("MAE  (Mean Absolute Error)     : {:.4f}".format(mae))
print("R2   (R-Squared)               : {:.4f}".format(r2))
print("========================================\n")

# --- 8. Simpan Model untuk Production dan Local ---
print("[7] Menyimpan Model ke HDFS dan Local Container...")

# 1. Simpan ke HDFS (Untuk Production/Sistem Terdistribusi)
hdfs_path = "hdfs://namenode3:9000/models/linear_reg"
model.write().overwrite().save(hdfs_path)
print("✅ Model tersimpan di HDFS: {}".format(hdfs_path))

# 2. Simpan ke Local File System di dalam Container Spark
local_path = "file:///opt/spark/models/linear_reg"
model.write().overwrite().save(local_path)
print("✅ Model tersimpan di Local Container: {}".format("/opt/spark/models/linreg_baseline_model"))

spark.stop()