import os
from pyspark.sql import SparkSession
# PENTING: Tambahkan import untuk fungsi matematika agregasi
from pyspark.sql.functions import col, lead, lag, when, mean as _mean, max as _max, min as _min, stddev as _stddev, sum as _sum
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from dotenv import load_dotenv

load_dotenv()

# --- 1. Inisialisasi Spark Session ---
print("[1] Memulai Spark Session...")
spark = SparkSession.builder \
    .appName("Bitcoin_LogReg_Advanced") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate() 

spark.sparkContext.setLogLevel("WARN")

# --- 2. Mengambil Data dari ClickHouse via JDBC ---
print("[2] Mengambil data dari ClickHouse (Dibatasi untuk Keamanan RAM)...")
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(os.getenv("CLICKHOUSE_DB", "bigdata"))
ch_user = os.getenv('CLICKHOUSE_USER', 'default')
ch_password = os.getenv('CLICKHOUSE_PASSWORD', '')

# Menambahkan time_unix dan membatasi data agar training berjalan cepat
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

# --- 3. Feature Engineering NATIVE Spark & Targeting (n+60) ---
print("[3] Melakukan Rekayasa Fitur (n-60) & Target (n+60)...")

# Mempartisi data per hari untuk menghindari "Moving all data to a single partition"
df = df.withColumn("date_part", col("time").cast("date"))

# Definisi Window (Jendela) 60 Detik
window_exact = Window.partitionBy("date_part").orderBy("time")
window_past_60 = Window.partitionBy("date_part").orderBy("time").rowsBetween(-60, 0)

# A. Ekstraksi Fitur 60 Detik ke Belakang (SAMA PERSIS DENGAN features_eng.py)
df = df.withColumn("close_mean_60", _mean("close").over(window_past_60))
df = df.withColumn("close_max_60", _max("close").over(window_past_60))
df = df.withColumn("close_min_60", _min("close").over(window_past_60))
df = df.withColumn("close_std_60", _stddev("close").over(window_past_60))
df = df.withColumn("volume_sum_60", _sum("volume").over(window_past_60))

df = df.withColumn("close_lag_60", lag("close", 60).over(window_exact))
df = df.withColumn("close_delta_60", col("close") - col("close_lag_60"))

# B. Target 60 Detik ke Depan
df = df.withColumn("next_close_60", lead("close", 60).over(window_exact))
df = df.withColumn("target", when(col("next_close_60") > col("close"), 1).otherwise(0))

# Bersihkan data yang kosong akibat pergeseran Window
df = df.dropna()
df = df.drop("date_part", "close_lag_60", "next_close_60")

# Tampilkan 5 baris pertama (biasanya kolom lag n-60 akan Null)
df.show(5, truncate=False)

# Tampilkan 5 baris terakhir (biasanya kolom target n+60 akan Null)
df.tail(5)

# --- 4. Membagi Data (Time-Series Split) ---
print("[4] Membagi data secara kronologis...")

# Cek apakah DataFrame memiliki data
if df.count() == 0:
    raise ValueError("DataFrame kosong! Cek kembali proses Rekayasa Fitur di Langkah 3.")

quantiles = df.approxQuantile("time_unix", [0.8], 0.01)

# Pengaman tambahan jika approxQuantile tetap gagal
if not quantiles:
    raise ValueError("Gagal menghitung kuantil. Pastikan kolom 'time_unix' bertipe numerik dan tidak sepenuhnya Null.")

split_time_numeric = quantiles[0]

train_df = df.filter(col("time_unix") <= split_time_numeric)
test_df = df.filter(col("time_unix") > split_time_numeric)

# --- 5. Membangun ML Pipeline ---
print("[5] Membangun ML Pipeline...")
# Menambahkan fitur-fitur makro yang baru saja dibuat di atas
fitur_kolom = [
    'open', 'high', 'low', 'close', 'volume', 
    'close_lag_1', 'volume_lag_1', 'close_roll_mean_5', 'close_delta',
    'close_mean_60', 'close_max_60', 'close_min_60', 'close_std_60', 'volume_sum_60', 'close_delta_60'
]

assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="raw_features")
# Logistic Regression SANGAT membutuhkan scaler, pertahankan!
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=100)

pipeline = Pipeline(stages=[assembler, scaler, lr])

# --- 6. Melatih Model ---
print("[6] Melatih model Logistic Regression (Distributed)...")
model = pipeline.fit(train_df)

# --- 7. Evaluasi Model ---
print("[7] Melakukan evaluasi model...")
predictions = model.transform(test_df)

evaluator_multi = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction")
accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
f1_score = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})

evaluator_bin = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator_bin.evaluate(predictions)

print("\n=== HASIL EVALUASI LOGISTIC REGRESSION (ADVANCED FITUR) ===")
print("Akurasi   : {:.2f}%".format(accuracy * 100))
print("Precision : {:.4f}".format(precision))
print("Recall    : {:.4f}".format(recall))
print("F1-Score  : {:.4f}".format(f1_score))
print("AUC (ROC) : {:.4f}".format(auc))
print("===========================================================\n")

# --- 8. Simpan Model untuk Production ---
model_path = "hdfs://namenode3:9000/models/logreg_advanced_model"
model.write().overwrite().save(model_path)
print("Model berhasil disimpan ke HDFS: {}".format(model_path))

spark.stop()