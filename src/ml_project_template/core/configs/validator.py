from ml_project_template.core.errors import InvalidEnvVarError  # type: ignore
from dotenv import load_dotenv
from typing import Any
import os
import re

load_dotenv()


def validate_env_vars() -> dict[str, Any]:
    """
    Validates environment variables by a given pattern/rule for each variable and returns the valid vars. If any environment variable is not valid, it fails.
    """
    errors: list[str] = []
    validated_vars: dict[str, Any] = {}
    ENV_SCHEMA = {
        "APP_MODE": re.compile("^(dev|test|prod)$"),
        "MLFLOW_TRACKING_URI": re.compile(r"^sqlite:\/\/\/[^\s]+\.db$"),
        "MLFLOW_EXPERIMENT_NAME": re.compile("^[A-Za-z-]+$"),
        "MLFLOW_DB_NAME": re.compile(r"^.+\.db$"),
    }
    for var, pattern in ENV_SCHEMA.items():
        value = os.getenv(var)

        if value is None:
            errors.append(f"Missing required environment variable: {var}")
            continue

        if not pattern.match(value):
            errors.append(
                f"Invalid value for {var}: '{value}' does not match {pattern}"
            )
            continue
        validated_vars[var] = value
    if errors:
        raise InvalidEnvVarError("\n".join(errors))
    return validated_vars


ENVS = validate_env_vars()
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)  # adjust as needed
MLFLOW_DB = os.path.join(PROJECT_ROOT, ENVS["MLFLOW_DB_NAME"])
ARTIFACT_ROOT = os.path.join(PROJECT_ROOT, "mlruns")
