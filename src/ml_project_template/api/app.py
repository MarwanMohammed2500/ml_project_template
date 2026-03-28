from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.ml_project_template.api.routes import api
from src.ml_project_template.api.inference import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="title for your app", lifespan=lifespan
)  # if you want to disable docs or redoc set the arguments with those names to None

app.include_router(api.router)


@app.get("/health")
def health():
    """Health check"""
    return {"healthy": True}
