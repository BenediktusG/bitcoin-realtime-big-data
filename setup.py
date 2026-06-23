import clickhouse_connect
import time
import sys
import os
from dotenv import load_dotenv
from confluent_kafka.admin import AdminClient, NewTopic

load_dotenv()



def setup_clickhouse():
    CH_HOST = os.getenv('CH_HOST', 'localhost')
    CH_PORT = os.getenv('CH_PORT', '8125')
    CH_USER = os.getenv('CLICKHOUSE_USER', 'default')
    CH_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')
    print("[*] Mencoba terhubung ke ClickHouse...")
    
    # Mekanisme retry untuk memastikan container sudah siap
    client = None
    for i in range(10):
        try:
            client = clickhouse_connect.get_client(
                host=CH_HOST, 
                port=CH_PORT, 
                username=CH_USER, 
                password=CH_PASSWORD
            )
            print("[+] Berhasil terhubung ke ClickHouse!")
            break
        except Exception:
            print(f"[-] Menunggu ClickHouse siap menerima koneksi HTTP (percobaan {i+1}/10)...")
            time.sleep(3)
            
    if not client:
        print("[!] Gagal terhubung. Pastikan 'docker-compose up -d' sudah dijalankan.")
        sys.exit(1)

    # Kumpulan perintah SQL (DDL & DCL)
    queries = [
        # 1. Setup Database
        "CREATE DATABASE IF NOT EXISTS bigdata;",
        
        # 2. Setup Spark User
        f"CREATE USER IF NOT EXISTS {os.getenv('SPARK_CH_USER')} IDENTIFIED WITH sha256_password BY '{os.getenv('SPARK_CH_PASSWORD')}';",
        "GRANT ALL ON bigdata.* TO spark_user;",
        
        # 3. Setup User Visualisasi (Grafana)
        f"CREATE USER IF NOT EXISTS {os.getenv('GRAFANA_CH_USER')} IDENTIFIED WITH sha256_password BY '{os.getenv('GRAFANA_CH_PASSWORD')}';",
        "GRANT SELECT ON bigdata.* TO grafana_user;",
        
        # 4. Setup Tabel dan Materialized View
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_orders (
            time DateTime,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_incoming_orders (
            time DateTime,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64
        ) ENGINE = Kafka()
        SETTINGS
            kafka_broker_list = 'kafka3:29092',
            kafka_topic_list = 'bitcoin-orders',
            kafka_group_name = 'clickhouse-consumer',
            kafka_format = 'JSONEachRow',
            kafka_num_consumers = 2;
        """,
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_mv TO bigdata.bitcoin_orders AS
        SELECT
            time,
            open,
            high,
            low,
            close,
            volume
        FROM bigdata.bitcoin_incoming_orders;
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_realtime_predictions (
            time DateTime,
            close_prediction Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_features (
            time DateTime,
            close Float64,
            volume Float64,
            volume_lag_1 Float64,
            close_delta Float64,
            dist_to_mean_5 Float64,
            dist_to_mean_60 Float64,
            dist_to_max_60 Float64,
            dist_to_min_60 Float64,
            close_std_60 Float64,
            volume_sum_60 Float64,
            close_delta_60 Float64
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions (
            time DateTime,
            close_prediction Nullable(Float64)
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_orders_1h_agg (
            time DateTime,
            open_state AggregateFunction(argMin, Float64, DateTime),
            high_state AggregateFunction(max, Float64),
            low_state AggregateFunction(min, Float64),
            close_state AggregateFunction(argMax, Float64, DateTime),
            avg_close_state AggregateFunction(avg, Float64),
            volume_state AggregateFunction(sum, Float64)
        ) ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_orders_1h_mv
        TO bigdata.bitcoin_orders_1h_agg AS
        SELECT
            toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
            argMinState(open, time) AS open_state,
            maxState(high) AS high_state,
            minState(low) AS low_state,
            argMaxState(close, time) AS close_state,
            avgState(close) AS avg_close_state,
            sumState(volume) AS volume_state
        FROM bigdata.bitcoin_orders
        GROUP BY time;
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions_1h_agg (
            time DateTime,
            close_prediction_state AggregateFunction(avg, Float64)
        ) ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_predictions_1h_mv
        TO bigdata.bitcoin_predictions_1h_agg AS
        SELECT
            toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
            avgState(assumeNotNull(close_prediction)) AS close_prediction_state
        FROM bigdata.bitcoin_predictions
        WHERE close_prediction IS NOT NULL
        GROUP BY time;
        """,
        f"""
        CREATE TABLE IF NOT EXISTS bigdata.bitcoin_features_1h_agg (
            time DateTime,
            close_delta_state AggregateFunction(avg, Float64),
            close_std_60_state AggregateFunction(avg, Float64),
            dist_to_mean_60_state AggregateFunction(avg, Float64),
            volume_sum_60_state AggregateFunction(avg, Float64)
        ) ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(time)
        ORDER BY (time);
        """,
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_features_1h_mv
        TO bigdata.bitcoin_features_1h_agg AS
        SELECT
            toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
            avgState(close_delta) AS close_delta_state,
            avgState(close_std_60) AS close_std_60_state,
            avgState(dist_to_mean_60) AS dist_to_mean_60_state,
            avgState(volume_sum_60) AS volume_sum_60_state
        FROM bigdata.bitcoin_features
        GROUP BY time;
        """,
        f"""
        INSERT INTO bigdata.bitcoin_orders_1h_agg
        SELECT
            toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
            argMinState(open, time) AS open_state,
            maxState(high) AS high_state,
            minState(low) AS low_state,
            argMaxState(close, time) AS close_state,
            avgState(close) AS avg_close_state,
            sumState(volume) AS volume_state
        FROM bigdata.bitcoin_orders
        WHERE (SELECT count() FROM bigdata.bitcoin_orders_1h_agg) = 0
        GROUP BY time;
        """,
        f"""
        INSERT INTO bigdata.bitcoin_predictions_1h_agg
        SELECT
            toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
            avgState(assumeNotNull(close_prediction)) AS close_prediction_state
        FROM bigdata.bitcoin_predictions
        WHERE close_prediction IS NOT NULL
          AND (SELECT count() FROM bigdata.bitcoin_predictions_1h_agg) = 0
        GROUP BY time;
        """,
        f"""
        INSERT INTO bigdata.bitcoin_features_1h_agg
        SELECT
            toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
            avgState(close_delta) AS close_delta_state,
            avgState(close_std_60) AS close_std_60_state,
            avgState(dist_to_mean_60) AS dist_to_mean_60_state,
            avgState(volume_sum_60) AS volume_sum_60_state
        FROM bigdata.bitcoin_features
        WHERE (SELECT count() FROM bigdata.bitcoin_features_1h_agg) = 0
        GROUP BY time;
        """
    ]

    print("[*] Mengeksekusi skema RBAC dan Tabel...")
    for i, query in enumerate(queries):
        try:
            client.command(query)
            # Menampilkan log potongan query yang berhasil dieksekusi
            snippet = query.splitlines()[1 if i > 4 else 0][:30].strip()
            print(f"  -> OK: {snippet}...")
        except Exception as e:
            print(f"  -> ERROR pada query: {str(e)}")

    print("[+] Setup infrastruktur ClickHouse selesai!")

def is_kafka_topic_exists(admin_client, topic_name):
    """Cek apakah topik Kafka sudah ada"""
    try:
        metadata = admin_client.list_topics(timeout=10)
        return topic_name in metadata.topics
    except Exception as e:
        print(f"[-] Gagal memeriksa topik Kafka: {str(e)}")
        return False
    
def create_kafka_topic(admin_client, topic_name, num_partitions=1, replication_factor=1):
    """Buat topik Kafka jika belum ada"""
    if is_kafka_topic_exists(admin_client, topic_name):
        print(f"[+] Topik Kafka '{topic_name}' sudah ada.")
        return
    
    topic = NewTopic(topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
    try:
        admin_client.create_topics([topic])
        print(f"[+] Topik Kafka '{topic_name}' berhasil dibuat.")
    except Exception as e:
        print(f"[-] Gagal membuat topik Kafka: {str(e)}")

def setup_kafka_topic():
    admin = AdminClient({
        'bootstrap.servers': os.getenv('KAFKA_URL')
    })
    create_kafka_topic(admin, 'bitcoin-orders')


if __name__ == '__main__':
    setup_kafka_topic()
    setup_clickhouse()
