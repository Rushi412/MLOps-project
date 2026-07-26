FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && python -m churn_service.training
EXPOSE 8000
CMD ["uvicorn", "churn_service.app:app", "--host", "0.0.0.0", "--port", "8000"]

