import os
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
# Kita cari batas waktu persentil 80% (secara kasar menggunakan perhitungan kuantil)
quantiles = df.approxQuantile("time", [0.8], 0.01) # Mencari waktu di batas 80%
split_time = quantiles[0]

train_df = df.filter(col("time") <= split_time)
test_df = df.filter(col("time") > split_time)

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

evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("\n=== HASIL EVALUASI SPARK BASELINE ===")

acc_percent = accuracy * 100
print("Akurasi: {:.2f}%\n".format(acc_percent))

# --- 8. Simpan Model untuk Production ---
model_path = "hdfs://namenode3:9000/models/logreg_baseline_model"
model.write().overwrite().save(model_path)
print("Model berhasil disimpan ke HDFS: {}".format(model_path))

spark.stop()