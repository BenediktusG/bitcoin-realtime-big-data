import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from dotenv import load_dotenv

# PENTING: Import modul LightGBM dari SynapseML
from synapse.ml.lightgbm import LightGBMClassifier

load_dotenv()

print("[1] Memulai Spark Session untuk LightGBM...")
spark = SparkSession.builder \
    .appName("Bitcoin_LightGBM_Advance") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate()

print("[2] Mengambil data dari ClickHouse...")
ch_url = f"jdbc:clickhouse://clickhouse3:8123/{os.getenv('CLICKHOUSE_DB', 'bigdata')}"
df = spark.read \
    .format("jdbc") \
    .option("url", ch_url) \
    .option("user", os.getenv('CLICKHOUSE_USER', 'default')) \
    .option("password", os.getenv('CLICKHOUSE_PASSWORD', '')) \
    .option("dbtable", "bitcoin_features") \
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .load()

print("[3] Mempersiapkan variabel target...")
windowSpec = Window.orderBy("time")
df = df.withColumn("next_close", lead("close", 1).over(windowSpec))
df = df.withColumn("target", when(col("next_close") > col("close"), 1).otherwise(0))
df = df.dropna()

print("[4] Membagi data secara kronologis (80% Train, 20% Test)...")
quantiles = df.approxQuantile("time", [0.8], 0.01)
split_time = quantiles[0]

train_df = df.filter(col("time") <= split_time)
test_df = df.filter(col("time") > split_time)

print("[5] Membangun ML Pipeline LightGBM...")
fitur_kolom = [
    'open', 'high', 'low', 'close', 'volume', 
    'close_lag_1', 'volume_lag_1', 'close_roll_mean_5', 'close_delta'
]

# Menggabungkan fitur
assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="features")

# Konfigurasi LightGBM
# Perhatikan bahwa kita tidak butuh StandardScaler lagi karena LightGBM berbasis Tree (tidak sensitif terhadap skala)
lgbm = LightGBMClassifier(
    objective="binary",
    featuresCol="features",
    labelCol="target",
    numIterations=100,      # Setara dengan n_estimators
    learningRate=0.05,
    maxDepth=5,
    isUnbalance=True        # Sangat berguna jika persentase naik/turun tidak seimbang
)

pipeline = Pipeline(stages=[assembler, lgbm])

print("[6] Melatih model LightGBM (Distributed)...")
model = pipeline.fit(train_df)

print("[7] Mengevaluasi model LightGBM...")
predictions = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"\n=== HASIL EVALUASI LIGHTGBM ADVANCE ===")
print(f"Akurasi: {accuracy * 100:.2f}%\n")

# Simpan Model ke HDFS
model_path = "hdfs://namenode3:9000/models/lgbm_advance_model"
model.write().overwrite().save(model_path)
print(f"Model berhasil disimpan ke HDFS: {model_path}")

spark.stop()