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

## If u wanna re-train tf model

### 7. Copy the clickhouse-jdbc to the container

```bash
docker cp clickhouse-jdbc.jar spark-master3:/clickhouse-jdbc.jar
```

### 8. install some dependencies for run the model script

install dotenv

```bash
docker exec -it spark-master3 pip install python-dotenv
```

install numpy

```bash
docker exec -it spark-master3 pip install numpy
```

### 9. Copy model_linear_reg.py to docker container:

```bash
docker cp model_linear_reg.py spark-master3:/model_linear_reg.py
```

### 10. do same shit to random_forest_reg.py

```bash
docker cp model_random_forest_reg.py spark-master3:/model_random_forest_reg.py
```

### 11. Run the script in docker container

change the file if u want to change the model. this command use 4gb ram and driver memory 2gb. it takes like 30-45 minutes to train. So go goon first.

```bash
docker exec -e CLICKHOUSE_USER="default" -e CLICKHOUSE_PASSWORD="SuperSecureAdminPassword123!" -it spark-master3 /opt/spark/bin/spark-submit --master spark://spark-master3:7077 --driver-memory 2g  --executor-memory 4g --jars /clickhouse-jdbc.jar --driver-class-path /clickhouse-jdbc.jar /model_linear_reg.py
```

### 10. The model we r train, located to container. So, copy that result to ur own laptop

```bash
docker cp nama_container_spark_kamu:/opt/spark/models/linreg_baseline_model ./linreg_baseline_model
```
