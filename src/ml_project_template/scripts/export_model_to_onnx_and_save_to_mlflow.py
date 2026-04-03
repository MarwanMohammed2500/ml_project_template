import click
import mlflow
from typing import Optional
import torch
import onnx
from mlflow.models.signature import infer_signature  # type: ignore
import pandas as pd
import os
import io

from ml_project_template.core.configs.validator import ENVS  # type: ignore


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
    assert mlflow.get_tracking_uri() == f"sqlite:///{ENVS['MLFLOW_DB_NAME']}", (
        "MLflow tracking URI is not set correctly!"
    )
    assert (
        mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME", ""))
        is not None
    ), "MLflow experiment is not set correctly!"

    input_sample = torch.randn(1, input_dim)

    model_to_export: torch.nn.Module
    if model_path is not None:
        model_to_export = torch.load(model_path)
    elif model_uri is not None:
        model_to_export = mlflow.pytorch.load_model(model_uri)  # type: ignore
    else:
        raise ValueError("Either `model_path`, or `model_uri` should be provided.")

    model_to_export = model_to_export.cpu()  # type: ignore
    input_sample = input_sample.cpu()
    model_to_export.eval()  # type: ignore

    with io.BytesIO() as buffer:
        torch.onnx.export(  # type: ignore
            model_to_export,
            input_sample,  # type: ignore
            buffer,  # type: ignore
            dynamo=True,
            input_names=["x"],
            output_names=["output"],
            dynamic_shapes={"x": {0: "batch"}},
        )
        buffer.seek(0)
        model = onnx.load_model_from_string(buffer.read())

    dataframe = pd.read_csv(path_to_dataset)
    features: pd.DataFrame
    label: pd.Series
    features, label = dataframe.drop("label", axis=1), dataframe["label"]
    signature = infer_signature(features, label)

    mlflow.onnx.log_model(  # type: ignore
        model,
        name="onnx_model",
        registered_model_name="SimpleModel_ONNX",
        signature=signature,
    )


if __name__ == "__main__":
    cli.add_command(export_model_to_onnx_and_save_to_mlflow)
    cli()
