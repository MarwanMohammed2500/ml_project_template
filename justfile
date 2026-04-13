set dotenv-load := true

# Builds the development environment and run the application
build_and_run_dev:
    docker compose -f docker/docker-compose.dev.yml up --build -d

# Runs the development environment without rebuilding the images (useful when you just want to restart the containers)
run_dev:
    docker compose -f docker/docker-compose.dev.yml up -d

# Run the unit tests
test:
    uv run pytest tests/

# Stop and remove all containers, networks, and volumes. Also cleans pytest and ruff cache.
clean:
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    find . -type d -name "__pycache__" -exec rm -r {} +
    docker-compose -f docker/docker-compose.dev.yaml down -v

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

# Train the model, uses yaml_path to load yaml configurations sepecific for the model trainer
train yaml_path="configs/training_configs.yaml":
    uv run -m src.ml_project_template.training.run training-pipeline --yaml_path {{yaml_path}}

# Launch the Gunicorn server
serve port="8000" num_workers="4" timeout="120":
    gunicorn src.ml_project_template.serving.api.app:app \
     --workers {{num_workers}} \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:{{port}} \
     --preload \
     --timeout {{timeout}}
