import os
import time
import logging
import pandas as pd
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

def run_batch_injector():
    client = get_clickhouse_client()
    if not client:
        return

    csv_file_path = 'dataset_2025_2026.csv'
    
    if not os.path.exists(csv_file_path):
        logging.error(f"File {csv_file_path} tidak ditemukan!")
        return

    logging.info(f"Mulai membaca {csv_file_path}...")
    t_start = time.time()

    try:
        nama_kolom = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        # Ditambahkan chunksize agar memuat data per 500.000 baris secara bertahap
        chunk_iterator = pd.read_csv(
            csv_file_path, 
            header=None, 
            names=nama_kolom,
            usecols=[0, 1, 2, 3, 4, 5],
            chunksize=500000
        )

        for i, df in enumerate(chunk_iterator):
            t_chunk_start = time.time()
            
            # konversi timestamp dari milidetik ke datetime
            df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            
            # drop baris yang memiliki nilai time yang tidak valid (NaT)
            df = df.dropna(subset=['time'])

            total_baris = len(df)
            logging.info(f"Batch ke-{i+1} | Berhasil memuat {total_baris:,} baris. Memulai injeksi ke ClickHouse...")

            # eksekusi batch insert ke ClickHouse
            client.insert_df('bitcoin_orders', df)
            
            t_chunk_end = time.time()
            logging.info(f"Batch ke-{i+1} selesai dieksekusi dalam {t_chunk_end - t_chunk_start:.2f} detik.")
        
        t_end = time.time()
        logging.info(f">>> BATCH INSERT SUKSES dalam {t_end - t_start:.2f} detik <<<")

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat memproses data: {e}")
    finally:
        client.close()

if __name__ == '__main__':
    run_batch_injector()