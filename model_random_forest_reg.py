import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, lag, when, mean as _mean, max as _max, min as _min, stddev as _stddev, sum as _sum
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from dotenv import load_dotenv

# PENTING: Import komponen khusus Regresi dan Random Forest Regressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

load_dotenv()

# --- 1. Inisialisasi Spark Session ---
print("[1] Memulai Spark Session (Random Forest Regressor)...")
spark = SparkSession.builder \
    .appName("Bitcoin_RFReg_Forecast") \
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
WHERE time >= toDateTime('2026-05-01 00:00:00')
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

# --- 3. Feature Engineering & Target Regresi (n+60) ---
print("[3] Melakukan Rekayasa Fitur (n-60) & Target Kontinu (n+60)...")

df = df.withColumn("date_part", col("time").cast("date"))

# Jendela Waktu
window_exact = Window.partitionBy("date_part").orderBy("time")
window_past_60 = Window.partitionBy("date_part").orderBy("time").rowsBetween(-60, 0)

# A. Pembuatan Fitur Jendela Waktu 60 Detik ke Belakang
df = df.withColumn("close_mean_60", _mean("close").over(window_past_60))
df = df.withColumn("close_max_60", _max("close").over(window_past_60))
df = df.withColumn("close_min_60", _min("close").over(window_past_60))
df = df.withColumn("close_std_60", _stddev("close").over(window_past_60))
df = df.withColumn("volume_sum_60", _sum("volume").over(window_past_60))

df = df.withColumn("close_lag_60", lag("close", 60).over(window_exact))
df = df.withColumn("close_delta_60", col("close") - col("close_lag_60"))

# B. Target Regresi: Berapa nilai nominal selisih harga 60 detik ke depan
df = df.withColumn("next_close_60", lead("close", 60).over(window_exact))
df = df.withColumn("target", col("next_close_60") - col("close"))

# Bersihkan nilai Null akibat pergeseran Window
df = df.dropna()
df = df.drop("date_part", "close_lag_60", "next_close_60")

# --- 4. Membagi Data (Time-Series Split) ---
print("[4] Membagi data secara kronologis (80% Train, 20% Test)...")
quantiles = df.approxQuantile("time_unix", [0.8], 0.01)
split_time_numeric = quantiles[0]

train_df = df.filter(col("time_unix") <= split_time_numeric)
test_df = df.filter(col("time_unix") > split_time_numeric)

# --- 5. Membangun ML Pipeline ---
print("[5] Membangun ML Pipeline Random Forest Regressor...")
fitur_kolom = [
    'open', 'high', 'low', 'close', 'volume', 
    'close_lag_1', 'volume_lag_1', 'close_roll_mean_5', 'close_delta',
    'close_mean_60', 'close_max_60', 'close_min_60', 'close_std_60', 'volume_sum_60', 'close_delta_60'
]

assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="features")

# Catatan Teknis: Algoritma berbasis pohon (Tree-based) tidak sensitif terhadap skala data,
# sehingga komponen StandardScaler sengaja dilewati agar komputasi menghemat RAM 4GB kamu.
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="target",
    numTrees=75,          # 75 pohon memberikan kestabilan prediksi angka kontinu yang baik
    maxDepth=6,           # Mengizinkan struktur pohon mendeteksi korelasi non-linear yang dalam
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
print("Mean Absolute Error (MAE)      : {:.4f}  (Rata-rata tebakan harga meleset sekian USD)".format(mae))
print("R-Squared (R2)                 : {:.4f}  (Semakin mendekati 1.0 semakin akurat)".format(r2))
print("==============================================\n")

# --- 8. Simpan Model (Double-Storage Backups) ---
print("[8] Menyimpan Model ke HDFS dan Local Container...")

# Backup 1: Penyimpanan Terdistribusi Hadoop HDFS (Untuk Production)
hdfs_path = "hdfs://namenode3:9000/models/rf_regressor_model"
model.write().overwrite().save(hdfs_path)
print("✅ Model berhasil diamankan di HDFS: {}".format(hdfs_path))

# Backup 2: Penyimpanan Lokal File System di dalam Container Spark
local_path = "file:///opt/spark/models/rf_regressor_model"
model.write().overwrite().save(local_path)
print("✅ Model berhasil diamankan di Local Container: {}".format("/opt/spark/models/rf_regressor_model"))

spark.stop()