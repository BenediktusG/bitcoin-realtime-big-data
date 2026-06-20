import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from dotenv import load_dotenv

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

load_dotenv()

# --- 1. Inisialisasi Spark Session ---
print("[1] Memulai Spark Session (Random Forest Regressor)...")
spark = SparkSession.builder \
    .appName("Bitcoin_RFReg_Optimized") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate() 
spark.sparkContext.setLogLevel("WARN")

# --- 2. Mengambil Data dari ClickHouse via JDBC (Feature Engineering Pushdown) ---
print("[2 & 3] Mengambil data dan melakukan Rekayasa Fitur di ClickHouse...")
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(os.getenv("CLICKHOUSE_DB", "bigdata"))
ch_user = os.getenv('CLICKHOUSE_USER', 'default')
ch_password = os.getenv('CLICKHOUSE_PASSWORD', '')

query = """
(
    -- Membuang kolom sementara/absolut yang tidak perlu (Setara dengan df.drop)
    SELECT * EXCEPT(date_part, close_lag_60, next_close_60, close_mean_60, close_max_60, close_min_60)
    FROM (
        SELECT 
            *,
            (close - close_lag_60) AS close_delta_60,
            (close - close_mean_60) / close_mean_60 AS dist_to_mean_60,
            (close_max_60 - close) / close AS dist_to_max_60,
            (close - close_min_60) / close_min_60 AS dist_to_min_60,
            (close - close_roll_mean_5) / close_roll_mean_5 AS dist_to_mean_5,
            (next_close_60 - close) AS target
        FROM (
            SELECT 
                *,
                toUnixTimestamp(time) AS time_unix,
                toDate(time) AS date_part,

                -- A. Pembuatan Fitur Jendela Waktu 60 Detik
                avg(close) OVER w_past_60 AS close_mean_60,
                max(close) OVER w_past_60 AS close_max_60,
                min(close) OVER w_past_60 AS close_min_60,
                stddevSamp(close) OVER w_past_60 AS close_std_60,
                sum(volume) OVER w_past_60 AS volume_sum_60,
                lag(close, 60) OVER w_exact AS close_lag_60,

                -- B. Target Regresi
                lead(close, 60) OVER w_exact AS next_close_60
                
            FROM bitcoin_features
            WHERE time >= toDateTime('2025-01-01 00:00:00')
            WINDOW 
                w_exact AS (PARTITION BY toDate(time) ORDER BY toUnixTimestamp(time)),
                w_past_60 AS (PARTITION BY toDate(time) ORDER BY toUnixTimestamp(time) ROWS BETWEEN 60 PRECEDING AND CURRENT ROW)
        )
        -- Setara dengan df.dropna()
        WHERE close_mean_60 IS NOT NULL 
          AND close_std_60 IS NOT NULL 
          AND close_lag_60 IS NOT NULL 
          AND next_close_60 IS NOT NULL
          AND close_roll_mean_5 IS NOT NULL
    )
) AS btc_data
"""

df = spark.read \
    .format("jdbc") \
    .option("url", ch_url) \
    .option("user", ch_user) \
    .option("password", ch_password) \
    .option("dbtable", query) \
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .option("partitionColumn", "time_unix") \
    .option("lowerBound", "1735689600") \
    .option("upperBound", "1775839739") \
    .option("numPartitions", "8") \
    .load()

# OPTIMASI: Kunci hasil tarikan data di memori
df.cache()

# --- 4. Membagi Data (Time-Series Split) ---
print("[4] Membagi data secara kronologis (80% Train, 20% Test)...")
quantiles = df.approxQuantile("time_unix", [0.8], 0.01)
split_time_numeric = quantiles[0]

train_df = df.filter(col("time_unix") <= split_time_numeric)
test_df = df.filter(col("time_unix") > split_time_numeric)

# --- 5. Membangun ML Pipeline ---
print("[5] Membangun ML Pipeline Random Forest Regressor...")
fitur_kolom = [
    'volume', 'volume_lag_1', 'close_delta', 'dist_to_mean_5', 
    'dist_to_mean_60', 'dist_to_max_60', 'dist_to_min_60', 
    'close_std_60', 'volume_sum_60', 'close_delta_60'
]

assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="features")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="target",
    numTrees=75,          
    maxDepth=6,           
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])

# --- 6. Melatih Model ---
print("[6] Melatih model Random Forest Regressor (Distributed)...")
model = pipeline.fit(train_df)

# --- 7. Evaluasi Model ---
print("[7] Melakukan evaluasi model regresi...")
predictions = model.transform(test_df)

evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction")

rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print("\n=== HASIL EVALUASI RANDOM FOREST REGRESSOR ===")
print("Root Mean Squared Error (RMSE) : {:.4f}".format(rmse))
print("Mean Absolute Error (MAE)      : {:.4f}".format(mae))
print("R-Squared (R2)                 : {:.4f}".format(r2))
print("==============================================\n")

# --- 8. Simpan Model (Double-Storage Backups) ---
print("[8] Menyimpan Model ke HDFS dan Local Container...")

hdfs_path = "hdfs://namenode3:9000/models/random_forest_reg"
model.write().overwrite().save(hdfs_path)
print("✅ Model berhasil diamankan di HDFS: {}".format(hdfs_path))

local_path = "file:///opt/spark/models/random_forest_reg"
model.write().overwrite().save(local_path)
print("✅ Model berhasil diamankan di Local Container: {}".format("/opt/spark/models/rf_regressor_optim_model"))

# Hapus cache untuk membersihkan RAM
df.unpersist()
spark.stop()