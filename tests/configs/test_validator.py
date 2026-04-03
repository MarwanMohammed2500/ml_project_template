from ml_project_template.configs import validate_env_vars  # type: ignore
import os


def test_validate_env_vars():
    os.environ["APP_ENV"] = "test"
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "experiment-name"
    os.environ["MLFLOW_DB_NAME"] = "mlflow.db"
    
    validated_vars = validate_env_vars()
    
    assert validated_vars["APP_ENV"] == "test"
    assert validated_vars["MLFLOW_TRACKING_URI"] == "sqlite:///mlflow.db"
    assert validated_vars["MLFLOW_EXPERIMENT_NAME"] == "experiment-name"
    assert validated_vars["MLFLOW_DB_NAME"] == "mlflow.db"
