from contextlib import asynccontextmanager
from fastapi import FastAPI
from ml_project_template.serving.api.routes import api  # type: ignore
from ml_project_template.serving.services import load_model  # type: ignore
from ml_project_template.core.logging import setup_logging  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="title for your app", lifespan=lifespan
)  # if you want to disable docs or redoc set the arguments with those names to None

app.include_router(api.router)

setup_logging(level="INFO", json_logs=True, output_file="serving_logs.log")


@app.get("/health")
def health():
    """Health check"""
    return {"healthy": True}
