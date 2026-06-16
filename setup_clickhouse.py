import clickhouse_connect
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi koneksi ke ClickHouse (via host port 8125)
CH_HOST = os.getenv('CH_HOST', 'localhost')
CH_PORT = os.getenv('CH_PORT', '8125')
CH_USER = os.getenv('CLICKHOUSE_USER', 'default')
CH_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')
print(f"[*] Konfigurasi ClickHouse: Host={CH_HOST}, Port={CH_PORT}, User={CH_USER} CH_PASSWORD={'***' if CH_PASSWORD else 'None'}")

def run_setup():
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
        "GRANT INSERT, SELECT ON bigdata.* TO spark_user;",
        
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
            kafka_broker_list = '{os.getenv("KAFKA_URL")}',
            kafka_topic_list = 'bitcoin-orders',
            kafka_group_name = 'clickhouse-consumer',
            kafka_format = 'JSONEachRow',
            kafka_num_consumers = 2;
        """,
        f"""
        CREATE MATERIALIZED VIEW iF NOT EXISTS bigdata.bitcoin_mv TO bigdata.bitcoin_orders AS
        SELECT
            time,
            open,
            high,
            low,
            close,
            volume
        FROM bigdata.bitcoin_incoming_orders;
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

if __name__ == '__main__':
    run_setup()