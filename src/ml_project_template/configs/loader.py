from src.ml_project_template.models import AppConfigs
from pydantic_yaml import parse_yaml_file_as


def load_configs(yaml_path: str):
    return parse_yaml_file_as(AppConfigs, yaml_path)