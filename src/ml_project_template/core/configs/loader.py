from ml_project_template.core.schemas import AppConfigs  # type: ignore
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as
from typing import TypeVar

OutputSchema = TypeVar("OutputSchema", bound=BaseModel)


def load_configs(
    yaml_path: str, configs_schema: type[OutputSchema] = AppConfigs
) -> OutputSchema:
    return parse_yaml_file_as(configs_schema, yaml_path)
