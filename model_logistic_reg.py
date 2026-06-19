import os
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from dotenv import load_dotenv

load_dotenv()

# --- 1. Inisialisasi Spark Session ---
print("[1] Memulai Spark Session...")
spark = SparkSession.builder \
    .appName("Bitcoin_LogReg_Baseline") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate() # Pastikan jar ClickHouse JDBC sudah ditambahkan saat submit

spark.sparkContext.setLogLevel("WARN")

# --- 2. Mengambil Data dari ClickHouse via JDBC ---
print("[2] Mengambil data dari ClickHouse...")
# Sesuaikan kredensial dengan environment kamu
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(
    os.getenv("CLICKHOUSE_DB", "bigdata")
)
ch_user = os.getenv('CLICKHOUSE_USER', 'default')
ch_password = os.getenv('CLICKHOUSE_PASSWORD', '')

df = spark.read \
    .format("jdbc") \
    .option("url", ch_url) \
    .option("user", ch_user) \
    .option("password", ch_password) \
    .option("dbtable", "bitcoin_features") \
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .load()

# --- 3. Mempersiapkan Variabel Target (Y) ---
print("[3] Mempersiapkan variabel target...")
# Menggunakan Window function Spark untuk melihat harga 'close' di baris selanjutnya (masa depan)
windowSpec = Window.orderBy("time")
df = df.withColumn("next_close", lead("close", 1).over(windowSpec))

# Target: 1 jika next_close > close, sisanya 0
df = df.withColumn("target", when(col("next_close") > col("close"), 1).otherwise(0))
df = df.dropna()

# --- 4. Membagi Data (Time-Series Split) ---
print("[4] Membagi data secara kronologis...")
# Di Spark, split waktu terbaik menggunakan filter timestamp, bukan randomSplit()

# 1. Ubah sementara kolom 'time' menjadi tipe angka (long/detik) untuk perhitungan kuantil
df_numeric_time = df.withColumn("time_long", col("time").cast("long"))

# 2. Cari batas waktu persentil 80% menggunakan kolom numerik tersebut
quantiles = df_numeric_time.approxQuantile("time_long", [0.8], 0.01) 
split_time_numeric = quantiles[0]

# 3. Bagi data menggunakan kolom 'time' yang di-cast ke long untuk perbandingan batas
train_df = df.filter(col("time").cast("long") <= split_time_numeric)
test_df = df.filter(col("time").cast("long") > split_time_numeric)

# --- 5. Membangun ML Pipeline ---
print("[5] Membangun ML Pipeline...")
fitur_kolom = [
    'open', 'high', 'low', 'close', 'volume', 
    'close_lag_1', 'volume_lag_1', 'close_roll_mean_5', 'close_delta'
]

# Spark ML membutuhkan semua fitur digabung menjadi 1 kolom 'features'
assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=100)

pipeline = Pipeline(stages=[assembler, scaler, lr])

# --- 6. Melatih Model ---
print("[6] Melatih model Logistic Regression (Distributed)...")
model = pipeline.fit(train_df)

# --- 7. Evaluasi Model ---
print("[7] Melakukan evaluasi model...")
predictions = model.transform(test_df)

# 1. Evaluator untuk Akurasi, Precision, Recall, dan F1-Score
evaluator_multi = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction")

accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
f1_score = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})

# 2. Evaluator Khusus Binary Classification untuk skor AUC
evaluator_bin = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator_bin.evaluate(predictions)

# Menampilkan Hasil Metrik
print("\n=== HASIL EVALUASI SPARK BASELINE ===")
print("Akurasi   : {:.2f}%".format(accuracy * 100))
print("Precision : {:.4f}  (Seberapa akurat tebakan 'Harga Naik' dari model)".format(precision))
print("Recall    : {:.4f}  (Berapa banyak momen 'Harga Naik' yang berhasil ditangkap)".format(recall))
print("F1-Score  : {:.4f}  (Keseimbangan antara Precision dan Recall)".format(f1_score))
print("AUC (ROC) : {:.4f}  (Kemampuan model membedakan kelas 1 dan 0, skor 0.5 = kebetulan)".format(auc))
print("=======================================\n")

# --- 8. Simpan Model untuk Production ---
model_path = "hdfs://namenode3:9000/models/logreg_baseline_model"
model.write().overwrite().save(model_path)
print("Model berhasil disimpan ke HDFS: {}".format(model_path))

spark.stop()