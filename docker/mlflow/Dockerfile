FROM ghcr.io/mlflow/mlflow:v2.16.2

# MLflow's base image does not ship the PostgreSQL DBAPI driver required by
# the platform's PostgreSQL backend store.
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.35.24
