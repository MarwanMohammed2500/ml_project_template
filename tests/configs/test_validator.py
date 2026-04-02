from src.ml_project_template.configs import validate_env_vars

def test_validate_env_vars():
    os.environ["APP_ENV"] = "dev"
    validated_vars = validate_env_vars()
    assert validated_vars["APP_ENV"] == "dev"