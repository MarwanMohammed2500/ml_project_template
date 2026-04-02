import uvicorn
from src.ml_project_template.configs import validate_env_vars

ENVS = validate_env_vars()

HOST = ENVS["HOST"]
PORT = ENVS["PORT"]
RELOAD = ENVS["RELOAD"]

if __name__ == "__main__":
    uvicorn.run("ml_project_template.api.app:app", host=HOST, port=PORT, reload=RELOAD)
