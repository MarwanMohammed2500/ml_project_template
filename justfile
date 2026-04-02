set dotenv-load := true

# Build the development environment and run the application
dev:
    docker compose -f docker/docker-compose.dev.yaml up --build

# Run the unit tests
test:
    uv run pytest tests/

# Stop and remove all containers, networks, and volumes. And clean pytest cache.
clean:
    rm -rf .pytest_cache
    docker-compose down -v

# Run the linter and auto-formatter (ruff in this case)
lint:
    uv run ruff check --fix
    uv run ruff format

# Start the MLflow server with a SQLite backend store
start_mlflow:
    mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --port 5001 \
    --host 0.0.0.0

# Run the script that exports a logged PyTorch model to ONNX and log that model to the registery
release_new_onnx_version version:
    python src/ml_project_template/scripts/export_model_to_onnx_and_save_to_mlflow.py \
    export-model-to-onnx-and-save-to-mlflow \
    --model_uri "models:/SimpleModel/{{version}}" \
    --input_dim 2 \
    --path_to_dataset /Users/marwanmohammed/Codes/ml_project_template/data/raw/binary_rawdata.csv