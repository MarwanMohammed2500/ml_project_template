from .routes import api
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .inference import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="title for your app",
    lifespan=lifespan
)  # if you want to disable docs or redoc set the arguments with those names to None

app.include_router(api.router)


@app.get("/health")
def health():
    return {"healthy": True}
