import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from dotenv import load_dotenv

# PENTING: Import Random Forest bawaan Spark
from pyspark.ml.classification import RandomForestClassifier

load_dotenv()

print("[1] Memulai Spark Session...")
# Tidak perlu lagi .config("spark.jars.packages"...)
spark = SparkSession.builder \
    .appName("Bitcoin_RandomForest_Advance") \
    .config("spark.master", "spark://spark-master3:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("[2] Mengambil data dari ClickHouse...")
ch_url = "jdbc:clickhouse://clickhouse3:8123/{}".format(os.getenv("CLICKHOUSE_DB", "bigdata"))

query = """
(SELECT 
    *,
    toUnixTimestamp(time) as time_unix
FROM bitcoin_features
WHERE time >= toDateTime('2025-01-01 00:00:00') -- Sesuaikan dengan tahun data Anda
) as btc_data
"""

df = spark.read \
    .format("jdbc") \
    .option("url", ch_url) \
    .option("user", os.getenv('CLICKHOUSE_USER', 'default')) \
    .option("password", os.getenv('CLICKHOUSE_PASSWORD', '')) \
    .option("dbtable", query) \
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .option("partitionColumn", "time_unix") \
    .option("lowerBound", "1735689600") \
    .option("upperBound", "1775839739") \
    .option("numPartitions", "8") \
    .load()

print("[3] Menghitung variabel target time-series...")
df = df.withColumn("date_part", col("time").cast("date"))
windowSpec = Window.partitionBy("date_part").orderBy("time")

df = df.withColumn("next_close", lead("close", 1).over(windowSpec))
df = df.withColumn("target", when(col("next_close") > col("close"), 1).otherwise(0))
df = df.dropna(subset=["next_close", "target"])
df = df.drop("date_part")

print("[4] Membagi data secara kronologis (80% Train, 20% Test)...")
quantiles = df.approxQuantile("time_unix", [0.8], 0.01) 
split_time_numeric = quantiles[0]

train_df = df.filter(col("time_unix") <= split_time_numeric)
test_df = df.filter(col("time_unix") > split_time_numeric)

print("[5] Membangun ML Pipeline Random Forest...")
fitur_kolom = [
    'open', 'high', 'low', 'close', 'volume', 
    'close_lag_1', 'volume_lag_1', 'close_roll_mean_5', 'close_delta'
]

assembler = VectorAssembler(inputCols=fitur_kolom, outputCol="features")

# Konfigurasi Random Forest Advance
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="target",
    numTrees=50,          # 50 pohon sudah cukup kuat sebagai awal
    maxDepth=5,           # Cegah overfitting
    featureSubsetStrategy="auto",
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])

print("[6] Melatih model Random Forest (Distributed)...")
model = pipeline.fit(train_df)

print("[7] Mengevaluasi model Random Forest...")
predictions = model.transform(test_df)

multi_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction")
accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
f1_score = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})

binary_evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_roc = binary_evaluator.evaluate(predictions)

print("\n=== HASIL EVALUASI RANDOM FOREST ADVANCE ===")
print("Akurasi   : {:.2f}%".format(accuracy * 100))
print("Precision : {:.4f}".format(precision))
print("Recall    : {:.4f}".format(recall))
print("F1-Score  : {:.4f}".format(f1_score))
print("AUC (ROC) : {:.4f}".format(auc_roc))
print("============================================\n")

print("[8] Menyimpan model ke HDFS...")
model_path = "hdfs://namenode3:9000/models/rf_advance_model"
model.write().overwrite().save(model_path)
print(f"Model berhasil disimpan ke HDFS: {model_path}")

spark.stop()