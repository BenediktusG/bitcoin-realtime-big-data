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

# --- 2. Mengambil Data dari ClickHouse via JDBC ---
print("[2] Mengambil data dari ClickHouse (Dibatasi untuk Keamanan RAM)...")
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(os.getenv("CLICKHOUSE_DB", "bigdata"))
ch_user = os.getenv('CLICKHOUSE_USER', 'default')
ch_password = os.getenv('CLICKHOUSE_PASSWORD', '')

query = """
(SELECT 
    *,
    toUnixTimestamp(time) as time_unix
FROM bitcoin_features
WHERE time >= toDateTime('2025-01-01 00:00:00')
) as btc_data
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

# --- 3. Pembersihan Data ---
print("[3] Membersihkan data dari Null...")
# Fitur dan target diasumsikan sudah ada dari ClickHouse (termasuk kolom 'target')
df = df.dropna()

# OPTIMASI: Kunci hasil pengambilan data di memori
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

# --- 8. Simpan Model (Double-Storage Backups Idempotent) ---
print("[8] Menyimpan Model ke HDFS dan Local Container (Mengganti model lama jika ada)...")

# Fungsi .overwrite() memastikan ini bersifat idempoten (menghapus yang lama, menyimpan yang baru)
hdfs_path = "hdfs://namenode3:9000/models/random_forest_reg"
model.write().overwrite().save(hdfs_path)
print("✅ Model baru berhasil diamankan di HDFS: {}".format(hdfs_path))

local_path = "file:///opt/spark/models/random_forest_reg"
model.write().overwrite().save(local_path)
print("✅ Model baru berhasil diamankan di Local Container: {}".format(local_path))

# Hapus cache untuk membersihkan RAM
df.unpersist()
spark.stop()