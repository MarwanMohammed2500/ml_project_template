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