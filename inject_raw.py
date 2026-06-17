import os
import time
import logging
import requests
from datetime import datetime
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

BATCH_SIZE = 5

def get_clickhouse_client():
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=int(CH_PORT), 
            username=CH_USER, password=CH_PASSWORD, database=CH_DB
        )
        return client
    except Exception as e:
        logging.error(f"Koneksi ClickHouse gagal: {e}")
        return None

def fetch_binance_data():
    url = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()[0]
        
        # transformasi data mentah menjadi format yang sesuai untuk ClickHouse
        dt_object = datetime.fromtimestamp(data[0] / 1000.0)
        return [
            dt_object, 
            float(data[1]), # Open
            float(data[2]), # High
            float(data[3]), # Low
            float(data[4]), # Close
            float(data[5])  # Volume
        ]
    except Exception as e:
        logging.error(f"Gagal menarik data dari API: {e}")
        return None

def run_injector():
    """
    Fungsi Load: Menjalankan loop utama untuk mengambil dan menyimpan data.
    Menggunakan mekanisme Micro-Batching untuk efisiensi database.
    """
    client = get_clickhouse_client()
    if not client:
        return

    buffer_data = []
    logging.info("Memulai streaming data mentah BTCUSDT... Tekan Ctrl+C untuk berhenti.")
    
    try:
        while True:
            row = fetch_binance_data()
            if row:
                buffer_data.append(row)
                logging.info(f"Buffer [{len(buffer_data)}/{BATCH_SIZE}] | Harga Close: {row[4]}")

                # Micro-Batching: Kirim data ke DB jika buffer penuh
                if len(buffer_data) >= BATCH_SIZE:
                    client.insert(
                        'bitcoin_orders', 
                        buffer_data, 
                        column_names=['time', 'open', 'high', 'low', 'close', 'volume']
                    )
                    logging.info("+++ Batch Insert Berhasil +++")
                    buffer_data.clear()
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        # Graceful Shutdown: Menyelamatkan sisa data di buffer sebelum mati
        if buffer_data:
            logging.info(f"Menyelamatkan {len(buffer_data)} data tersisa ke database...")
            client.insert(
                'bitcoin_orders', 
                buffer_data, 
                column_names=['time', 'open', 'high', 'low', 'close', 'volume']
            )
        logging.info("Proses injeksi dihentikan dengan aman.")
    finally:
        client.close()

if __name__ == '__main__':
    run_injector()