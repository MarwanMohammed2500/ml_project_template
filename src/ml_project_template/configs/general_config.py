from src.ml_project_template.exceptions import InvalidEnvVarError
from src.ml_project_template.models import AppConfigs
from pydantic_yaml import parse_yaml_file_as
from dotenv import load_dotenv
import os
import re

load_dotenv()

##############################
# You can define your environment variables here and import them anywhere in the project.
# Either this, or you can define a configs.py file for each sub-module to keep things clean and modular, if that is needed, but only for loading env variables.
# Keeping load_configs here is better
##############################


def load_configs(yaml_path: str):
    return parse_yaml_file_as(AppConfigs, yaml_path)


def validate_env_vars():
    """
    Validates environment variables by a given pattern/rule for each variable and returns the valid vars. If any environment variable is not valid, it fails.
    """
    errors = []
    validated_vars = {}
    ENV_SCHEMA = {
        "APP_ENV": re.compile("^(development|staging|production)$"),
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
