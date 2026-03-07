from .routes import api
from fastapi import FastAPI

app = FastAPI(
    title="title for your app"
)  # if you want to disable docs or redoc set the arguments with those names to None

app.include_router(api.router)


@app.get("/health")
def health():
    return {"healthy": True}
