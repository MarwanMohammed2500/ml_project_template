from ml_project_template.core.configs import load_configs  # type: ignore
from ml_project_template.core.schemas import TrainingConfigs  # type: ignore
import torch
from torch import nn, optim
from .model import Trainer
from .utils import create_data_loader
from .early_stopping import EarlyStopping
from .utils import count_model_parameters
from ml_project_template.core.utils import PreprocessorPipeline, Normalizer  # type: ignore
from ml_project_template.core.utils import set_seed  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from ml_project_template.core.logging import setup_logging  # type: ignore
import click
import pandas as pd
import mlflow
from typing import Any
from mlflow.models.signature import infer_signature  # type: ignore
from torch.utils.data import DataLoader
import numpy as np
import tempfile
import pickle
import json
import onnx
import io
import os


@click.group()
def cli_trainer():
    """CLI tool for training your models"""
    pass


def read_dataset(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dataset_df = pd.read_csv(dataset_path)
    X, y = dataset_df.drop("label", axis=1), dataset_df["label"]
    return dataset_df, X, y


def create_train_and_test_dataloaders(
    train_test_splitted_data: list[Any], batch_size: int
) -> tuple[DataLoader[Any], DataLoader[Any]]:

    X_train, X_test, y_train, y_test = train_test_splitted_data
    train_dataloader = create_data_loader(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )  # type: ignore
    test_dataloader = create_data_loader(X_test, y_test, batch_size=batch_size)  # type: ignore
    return train_dataloader, test_dataloader


def log_run_to_mlflow(
    train_loss: float,
    test_loss: float,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    torch_logged_model_name: str,
    onnx_logged_model_name: str,
    h_params: dict[str, Any],
    example_input: torch.Tensor,
    dataset_df: pd.DataFrame,
    model: nn.Module,
    preproc_pipeline: PreprocessorPipeline,
):
    with mlflow.start_run(run_name="model_training_run"):
        mlflow.log_table(
            count_model_parameters(model=model, example_input=example_input),
            artifact_file="model_summary.json",
        )

        mlflow.log_params(h_params)

        dataset = mlflow.data.from_pandas(  # type: ignore
            dataset_df, name="train-mock-dataset", targets="label"
        )
        mlflow.log_input(dataset, context="training")  # type: ignore

        all_metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        mlflow.log_metrics(all_metrics)

        with io.BytesIO() as buffer:
            pickle.dump(preproc_pipeline, buffer)

            buffer.seek(0)

            with tempfile.TemporaryDirectory() as tmp_dir:
                pkl_path = os.path.join(tmp_dir, "preproc_pipeline.pkl")
                json_path = os.path.join(tmp_dir, "normalization_constants.json")

                norm_params = {
                    "norm_mean": h_params.get("norm_mean"),
                    "norm_std": h_params.get("norm_std"),
                }

                with open(pkl_path, "wb") as f:
                    pickle.dump(preproc_pipeline, f)
                with open(json_path, "w") as f:
                    json.dump(norm_params, f)

                model.eval()
                signature = infer_signature(
                    dataset_df, model(example_input).detach().cpu().numpy()
                )
                mlflow.pytorch.log_model(  # type: ignore
                    model,
                    name=torch_logged_model_name,
                    registered_model_name=model.__class__.__name__,
                    signature=signature,
                    input_example=dataset_df.drop("label", axis=1).iloc[[0]],  # type: ignore
                    extra_files=[pkl_path, json_path],
                )

                with io.BytesIO() as buffer:
                    torch.onnx.export(  # type: ignore
                        model,
                        example_input,  # type: ignore
                        buffer,  # type: ignore
                        dynamo=True,
                        input_names=["x"],
                        output_names=["output"],
                        dynamic_shapes={"x": {0: "batch"}},
                    )
                    buffer.seek(0)
                    onnx_model = onnx.load_model_from_string(buffer.read())
                mlflow.onnx.log_model(  # type: ignore
                    onnx_model,
                    name=onnx_logged_model_name,
                    registered_model_name=f"{model.__class__.__name__}_ONNX",
                    signature=signature,
                    input_example=dataset_df.drop("label", axis=1).iloc[[0]],
                    extra_files=[pkl_path, json_path],
                )


@click.command()
@click.option("--yaml_path", default=None, help="Path to your YAML trainer configs")
@click.option(
    "--torch_logged_model_name",
    default="torch_model",
    help="The name to use when logging the trained torch model",
)
@click.option(
    "--onnx_logged_model_name",
    default="onnx_model",
    help="The name to use when logging the exported onnx model",
)
def training_pipeline(
    yaml_path: str, torch_logged_model_name: str, onnx_logged_model_name: str
):
    set_seed()

    training_configs: TrainingConfigs = load_configs(
        yaml_path=yaml_path, configs_schema=TrainingConfigs
    )

    dataset_df, X, y = read_dataset(training_configs.data_configs.dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(  # type: ignore
        X, y, test_size=0.2, random_state=42
    )

    normalizer = Normalizer(
        mean=X_train.values.mean(axis=0),  # type: ignore
        std=X_train.values.std(axis=0),  # type: ignore
    )
    preproc_pipeline = PreprocessorPipeline(steps=[normalizer])
    processed_X_train = preproc_pipeline(X_train.values)  # type: ignore
    processed_X_test = preproc_pipeline(X_test.values)  # type: ignore

    batch_size = training_configs.trainer_configs.batch_size

    train_dataloader, test_dataloader = create_train_and_test_dataloaders(
        train_test_splitted_data=[
            processed_X_train,
            processed_X_test,
            y_train.values,  # type: ignore
            y_test.values,  # type: ignore
        ],
        batch_size=batch_size,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(
        patience=training_configs.trainer_configs.early_stopping_patience,
        delta=training_configs.trainer_configs.early_stopping_delta,
    )

    num_classes = 2

    lr_scheduler_kwargs: dict[str, Any] = {
        "step_size": training_configs.trainer_configs.lr_scheduler_step_size,
        "gamma": training_configs.trainer_configs.lr_scheduler_gamma,
    }

    trainer = Trainer(
        task_type=training_configs.trainer_configs.task_type,
        num_epochs=training_configs.trainer_configs.num_epochs,
        loss_fn=loss_fn,
        optimizer_class=optim.Adam,
        learning_rate=training_configs.trainer_configs.learning_rate,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        num_classes=num_classes,
        model_uri=training_configs.trainer_configs.model_uri,
        lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
        early_stopper=early_stopper,
        binary_decision_threshold=training_configs.trainer_configs.binary_decision_threshold,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    train_loss, test_loss, train_metrics, test_metrics = trainer.train()

    h_params: dict[str, Any] = {
        "learning_rate": training_configs.trainer_configs.learning_rate,
        "optimizer": trainer.optimizer_class,
        "loss_function": loss_fn.__class__.__name__,
        "num_epochs": training_configs.trainer_configs.num_epochs,
        "batch_size": batch_size,
        "early_stopping_patience": training_configs.trainer_configs.early_stopping_patience,
        "early_stopping_delta": training_configs.trainer_configs.early_stopping_delta,
        "binary_decision_threshold": training_configs.trainer_configs.binary_decision_threshold,
        "norm_mean": normalizer.mean.tolist()
        if isinstance(normalizer.mean, np.ndarray)
        else float(normalizer.mean),
        "norm_std": normalizer.std.tolist()
        if isinstance(normalizer.std, np.ndarray)
        else float(normalizer.std),
    }
    X_train_example, _ = next(iter(train_dataloader))
    model = trainer.model

    log_run_to_mlflow(
        train_loss=train_loss,
        test_loss=test_loss,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        torch_logged_model_name=torch_logged_model_name,
        onnx_logged_model_name=onnx_logged_model_name,
        h_params=h_params,
        example_input=X_train_example,
        dataset_df=dataset_df,
        model=model,
        preproc_pipeline=preproc_pipeline,
    )


if __name__ == "__main__":
    setup_logging(level="INFO", json_logs=True, output_file="./training_logs.log")
    cli_trainer.add_command(training_pipeline)
    cli_trainer()
