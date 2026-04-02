from pydantic import BaseModel
from typing import Literal, Optional


class TrainerConfigs(BaseModel):
    task_type: Literal["regression", "binary", "multiclass"]
    num_epochs: int
    model_path: Optional[str] = None
    model_uri: Optional[str] = None
    verbose: bool
    binary_decision_threshold: float


class DataConfigs(BaseModel):
    dataset_path: str


class TrainingConfigs(BaseModel):
    """
    Full system configurations

    ---
    Attributes:
        trainer_configs:
            Training configurations for the trainer.

        data_configs:
            Information about the data like where the dataset file is.
    """

    trainer_configs: TrainerConfigs
    data_configs: DataConfigs
