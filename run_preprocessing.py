import os
import time
import logging
from dotenv import load_dotenv
import clickhouse_connect

logging.basicConfig(
    level=logging.INFO, 
    format='[*] %(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

CH_HOST = os.getenv('CH_HOST', 'localhost')
CH_PORT = os.getenv('CH_PORT', '8125')
CH_USER = os.getenv('CLICKHOUSE_USER', 'default')
CH_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')
CH_DB = os.getenv('CLICKHOUSE_DB', 'bigdata')

# connect ke clickHouse
def get_clickhouse_client():
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD, database=CH_DB
        )
        return client
    except Exception as e:
        logging.error(f"Gagal terhubung ke ClickHouse: {e}")
        raise

def setup_features_table(client):
    # membuat table jika belum ada 
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS bigdata.bitcoin_features (
        time DateTime,
        open Float64,
        high Float64,
        low Float64,
        close Float64,
        volume Float64,
        close_lag_1 Float64,
        volume_lag_1 Float64,
        close_roll_mean_5 Float64,
        close_delta Float64
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(time)
    ORDER BY (time);
    """
    client.command(create_table_query)
    logging.info("Tabel 'bitcoin_features' siap digunakan.")


def run_preprocessing(client):
    logging.info("Memulai proses preprocessing (Lag, Rolling Mean, Delta)...")
    
    transform_query = """
    INSERT INTO bigdata.bitcoin_features
    SELECT 
        time, open, high, low, close, volume,
        lagInFrame(close, 1) OVER w AS close_lag_1,
        lagInFrame(volume, 1) OVER w AS volume_lag_1,
        avg(close) OVER w5 AS close_roll_mean_5,
        close - lagInFrame(close, 1) OVER w AS close_delta
    FROM bigdata.bitcoin_orders
    WHERE time > (
        SELECT coalesce(max(time), toDateTime('1970-01-01 00:00:00')) 
        FROM bigdata.bitcoin_features
    )
    WINDOW 
        w AS (ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
        w5 AS (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW);
    """

    t_start = time.time()
    client.command(transform_query)
    t_end = time.time()

    count = client.command("SELECT count() FROM bigdata.bitcoin_features")
    logging.info(f"Preprocessing selesai dalam {t_end - t_start:.2f} detik.")
    logging.info(f"Total akumulasi baris data matang saat ini: {count}")


if __name__ == '__main__':
    ch_client = get_clickhouse_client()
    
    if ch_client:
        # Inisialisasi infrastruktur
        setup_features_table(ch_client)
        
        # Eksekusi proses preprocessing
        run_preprocessing(ch_client)
        
        ch_client.close()