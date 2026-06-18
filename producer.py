import json
import time
import os
import random
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Producer

load_dotenv()

KAFKA_URL = os.getenv('KAFKA_URL', '127.0.0.1:9094')
TOPIC_NAME = 'bitcoin-orders'

def delivery_report(err, msg):
    if err is not None:
        print(f"[-] Gagal mengirim pesan: {err}")
    else:
        print(f"[+] Pesan terkiritopic_namem ke {msg.topic()} [{msg.partition()}] pada offset {msg.offset()}")

def generate_bitcoin_data(last_price):
    change_percent = random.uniform(-0.001, 0.001)  
    open_price = last_price
    close_price = open_price * (1 + change_percent)
    
    high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.0005))
    low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.0005))
    
    volume = random.uniform(0.1, 10.0)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    data = {
        "time": current_time,
        "open": round(open_price, 2),
        "high": round(high_price, 2),
        "low": round(low_price, 2),
        "close": round(close_price, 2),
        "volume": round(volume, 6)
    }
    return data, close_price

def main():
    print(f"[*] Menghubungkan ke Kafka broker di {KAFKA_URL}...")
    conf = {
        'bootstrap.servers': KAFKA_URL,
        'client.id': 'bitcoin-producer'
    }
    
    try:
        producer = Producer(conf)
        print("[+] Berhasil terhubung ke Kafka!")
    except Exception as e:
        print(f"[!] Gagal membuat produser Kafka: {e}")
        return

    print(f"[*] Mulai memproduksi data ke topik '{TOPIC_NAME}' (Tekan Ctrl+C untuk berhenti)...")
    
    current_price = 65000.00
    
    try:
        while True:
            data, current_price = generate_bitcoin_data(current_price)
            
            message_payload = json.dumps(data)
            
            producer.produce(
                topic=TOPIC_NAME,
                value=message_payload.encode('utf-8'),
                callback=delivery_report
            )
            
            producer.poll(0)
            
            print(f"  -> Data: {message_payload}")
            
            # Tunggu 1 detik
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Menghentikan produser...")
    finally:
        print("[*] Melakukan flushing sisa data di buffer...")
        producer.flush()
        print("[+] Produser berhasil dihentikan secara aman.")

if __name__ == '__main__':
    main()
