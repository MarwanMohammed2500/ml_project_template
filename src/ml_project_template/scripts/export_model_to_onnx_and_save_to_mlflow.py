import click
import mlflow
from typing import Optional
from ml_project_template.utils import export_onnx_from_torch  # type: ignore
import torch
import onnx
from mlflow.models.signature import infer_signature  # type: ignore
import pandas as pd
import os

from ml_project_template.configs.validator import ENVS  # type: ignore

@click.group()
def cli():
    """A handy CLI tool."""
    pass


@click.command()
@click.option(
    "--model_uri", default=None, help='MLFlow model URI: "models:/<model_name>/<stage>"'
)
@click.option(
    "--model_path", default=None, help="Local path to PyTorch model state_dict"
)
@click.option(
    "--input_dim", type=int, prompt=True, help="Input feature dimension for the model"
)
@click.option(
    "--path_to_dataset",
    type=str,
    prompt=True,
    help="Path to the CSV dataset used to train the model",
)
def export_model_to_onnx_and_save_to_mlflow(
    model_path: Optional[str],
    model_uri: Optional[str],
    input_dim: int,
    path_to_dataset: str,
) -> None:
    assert mlflow.get_tracking_uri() == f"sqlite:///{ENVS["MLFLOW_DB_NAME"]}", (
        "MLflow tracking URI is not set correctly!"
    )
    assert mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME", "")) is not None, (
        "MLflow experiment is not set correctly!"
    )
    input_sample = torch.randn(1, input_dim)
    output_path = "model.onnx"

    export_onnx_from_torch(
        input_sample=input_sample,
        output_path=output_path,
        input_names=["x"],
        output_names=["output"],
        dynamic_shapes={"x": {0: "batch"}},
        model_uri=model_uri,
        model_path=model_path,
    )
    model = onnx.load(output_path)  # type: ignore
    dataframe = pd.read_csv(path_to_dataset)
    features: pd.DataFrame
    label: pd.Series
    features, label = dataframe.drop("label", axis=1), dataframe["label"]
    signature = infer_signature(features, label)
    mlflow.onnx.log_model(  # type: ignore
        model,
        name="onnx_model",
        registered_model_name=f"SimpleModel_ONNX",
        signature=signature,
    )


if __name__ == "__main__":
    cli.add_command(export_model_to_onnx_and_save_to_mlflow)
    cli()
