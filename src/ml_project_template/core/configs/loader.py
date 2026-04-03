from ml_project_template.core.schemas import AppConfigs  # type: ignore
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as
from typing import TypeVar

OutputSchema = TypeVar("OutputSchema", bound=BaseModel)


def load_configs(
    yaml_path: str, configs_schema: type[OutputSchema] = AppConfigs
) -> OutputSchema:
    """Loads YAML configurations and matches them to proper PyDantic Model to ensure that the values are valid
    
    Args:
        yaml_path: str:
            The path to the configuration YAML file.
        
        configs_schema: type[OutputSchema] = AppConfigs:
            The PyDantic Configuration Schema to run the YAML configs against.
    
    OutputSchema = TypeVar("OutputSchema", bound=BaseModel)
    """
    return parse_yaml_file_as(configs_schema, yaml_path)
