# Bitcoin Real-time Big Data Pipeline

## Quick Start

### 1. Prerequisites
Ensure you have the following installed:
- Docker and Docker Compose
- Python 3.11.x

### 2. Configure Environment Variables
Copy the configuration template and fill in the required values:
```bash
cp env.txt .env
```

### 3. Start Infrastructure
Run all container services in the background:
```bash
docker compose up -d
```

### 4. Setup Python Virtual Environment and Dependencies
Create a virtual environment, activate it, and install the required Python libraries:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Initialize Database and Kafka Topics
Run the setup script to create the database schemas, tables, materialized views in ClickHouse, and the required Kafka topics:
```bash
python setup.py
```

### 6. Run the Data Producer
Start streaming simulated Bitcoin order data to the Kafka topic:
```bash
python producer.py
```
