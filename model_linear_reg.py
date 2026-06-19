import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, lag, when, mean as _mean, max as _max, min as _min, stddev as _stddev, sum as _sum
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from dotenv import load_dotenv

# PENTING: Import untuk Regresi
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

load_dotenv()

print("[1] Memulai Spark Session (Linear Regression)...")
spark = SparkSession.builder \
    .appName("Bitcoin_LinReg_Forecast") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate() 
spark.sparkContext.setLogLevel("WARN")

print("[2] Mengambil Data dari ClickHouse...")
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(os.getenv("CLICKHOUSE_DB", "bigdata"))

query = """
(SELECT *, toUnixTimestamp(time) as time_unix
 FROM bitcoin_features
 WHERE time >= toDateTime('2025-01-01 00:00:00')
) as btc_data
"""

df = spark.read.format("jdbc").option("url", ch_url) \
    .option("user", os.getenv('CLICKHOUSE_USER', 'default')) \
    .option("password", os.getenv('CLICKHOUSE_PASSWORD', '')) \
    .option("dbtable", query).option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .option("partitionColumn", "time_unix").option("lowerBound", "1735689600") \
    .option("upperBound", "1775839739").option("numPartitions", "8").load()

print("[3] Feature Engineering (n-60) & Target Regresi (n+60)...")
df = df.withColumn("date_part", col("time").cast("date"))

window_exact = Window.partitionBy("date_part").orderBy("time_unix")
window_past_60 = Window.partitionBy("date_part").orderBy("time_unix").rangeBetween(-60, 0)

# A. Fitur Masa Lalu
df = df.withColumn("close_mean_60", _mean("close").over(window_past_60))
df = df.withColumn("close_max_60", _max("close").over(window_past_60))
df = df.withColumn("close_min_60", _min("close").over(window_past_60))
df = df.withColumn("close_std_60", _stddev("close").over(window_past_60))
df = df.withColumn("volume_sum_60", _sum("volume").over(window_past_60))

df = df.withColumn("close_lag_60", lag("close", 60).over(window_exact))
df = df.withColumn("close_delta_60", col("close") - col("close_lag_60"))

df = df.withColumn("dist_to_mean_60", (col("close") - col("close_mean_60")) / col("close_mean_60"))
df = df.withColumn("dist_to_max_60", (col("close_max_60") - col("close")) / col("close"))
df = df.withColumn("dist_to_min_60", (col("close") - col("close_min_60")) / col("close_min_60"))

# B. TARGET REGRESI (Selisih harga masa depan dengan harga saat ini)
df = df.withColumn("next_close_60", lead("close", 60).over(window_exact))
df = df.withColumn("target", col("next_close_60") - col("close"))

df = df.dropna().drop("date_part", "close_lag_60", "next_close_60")

print("[4] Membagi Data (Time-Series Split)...")
quantiles = df.approxQuantile("time_unix", [0.8], 0.01)
split_time_numeric = quantiles[0]

train_df = df.filter(col("time_unix") <= split_time_numeric)
test_df = df.filter(col("time_unix") > split_time_numeric)

print("[5] Membangun ML Pipeline Regresi...")
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

print("[6] Melatih Model Linear Regression...")
model = pipeline.fit(train_df)

print("[7] Evaluasi Model...")
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
print("[8] Menyimpan Model ke HDFS dan Local Container...")

# 1. Simpan ke HDFS (Untuk Production/Sistem Terdistribusi)
hdfs_path = "hdfs://namenode3:9000/models/linreg_baseline_model"
model.write().overwrite().save(hdfs_path)
print("✅ Model tersimpan di HDFS: {}".format(hdfs_path))

# 2. Simpan ke Local File System di dalam Container Spark
# Gunakan awalan file:// agar tidak salah masuk ke HDFS
local_path = "file:///opt/spark/models/linreg_baseline_model"
model.write().overwrite().save(local_path)
print("✅ Model tersimpan di Local Container: {}".format("/opt/spark/models/linreg_baseline_model"))

spark.stop()