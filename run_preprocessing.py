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
    """
    client.command(create_table_query)
    logging.info("Tabel 'bitcoin_features' siap digunakan (ReplacingMergeTree).")


def run_preprocessing(client):
    logging.info("Memulai proses preprocessing untuk fitur Random Forest (Volume/Close Lag, Delta, Distances, StdDev, Sum)...")
    
    transform_query = """
    INSERT INTO bigdata.bitcoin_features
    WITH 
        (SELECT coalesce(max(time), toDateTime('1970-01-01 00:00:00')) FROM bigdata.bitcoin_features) AS max_time,
        (
            SELECT min(time) FROM (
                SELECT time FROM bigdata.bitcoin_orders 
                WHERE time <= max_time 
                ORDER BY time DESC 
                LIMIT 60
            )
        ) AS context_start_time,
        coalesce(context_start_time, toDateTime('1970-01-01 00:00:00')) AS start_time
    SELECT 
        time, close, volume,
        volume_lag_1,
        close_delta,
        dist_to_mean_5,
        dist_to_mean_60,
        dist_to_max_60,
        dist_to_min_60,
        close_std_60,
        volume_sum_60,
        close_delta_60
    FROM (
        SELECT 
            time, close, volume,
            lagInFrame(volume, 1) OVER w AS volume_lag_1,
            close - lagInFrame(close, 1) OVER w AS close_delta,
            (close - avg(close) OVER w5) / avg(close) OVER w5 AS dist_to_mean_5,
            (close - avg(close) OVER w60) / avg(close) OVER w60 AS dist_to_mean_60,
            (max(close) OVER w60 - close) / close AS dist_to_max_60,
            (close - min(close) OVER w60) / min(close) OVER w60 AS dist_to_min_60,
            stddevSamp(close) OVER w60 AS close_std_60,
            sum(volume) OVER w60 AS volume_sum_60,
            close - lagInFrame(close, 60) OVER w AS close_delta_60
        FROM bigdata.bitcoin_orders
        WHERE time >= start_time
        WINDOW 
            w AS (ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
            w5 AS (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
            w60 AS (ORDER BY time ROWS BETWEEN 60 PRECEDING AND CURRENT ROW)
    )
    WHERE time > max_time
    """

    t_start = time.time()
    client.command(transform_query)
    
    logging.info("Mengoptimalkan tabel bitcoin_features (Deduplikasi ReplacingMergeTree)...")
    client.command("OPTIMIZE TABLE bigdata.bitcoin_features FINAL")
    
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