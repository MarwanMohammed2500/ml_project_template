from src.ml_project_template.errors import InvalidModelPathError
from src.ml_project_template.metrics import (
    set_classification_metrics,
)  # add regression related ones when implementing regression training.
from typing import Literal, Optional, Any
from tqdm.auto import tqdm
import logging
import torch
import os
from torch.utils.data import DataLoader
from src.ml_project_template.early_stopping import EarlyStopping
import mlflow

logger = logging.getLogger(__name__)


class Trainer:
    """
    Base Trainer class. This acts as the main training engine, it takes care of the training loop, logging, and early stopping.
    And builds finer strategies from other subclasses which implements the step function for a certain task/type of model, and sets the appropriate metrics to log.

    ---
    Args:
        task_type: Literal["regression", "binary", "multiclass"]:
            The type of task the model is trained for, used to determine how it will be trained

        num_epochs: int:
            The number of epochs to train the model for

        loss_fn: torch.nn.modules.loss:
            The loss function to use

        optimizer: torch.optim:
            The optimizer to use

        train_dataloader: torch.utils.data.DataLoader:
            The training set in a PyTorch DataLoader

        test_dataloader: torch.utils.data.DataLoader:
            The test set in a PyTorch DataLoader

        model: Optional[torch.nn.Module] = None:
            The model to train, defaults to None Set when you are training a model from scratch, not finetuning weights.

        pretrained_model_path: Optional[str] = None:
            The path to the pretrained model weights to finetune, defaults to None.

        lr_scheduler: Optional[torch.optim.lr_scheduler] = None:
            The learning rate scheduler, defaults to None

        early_stopping_class: Optional[type] = None:
            Early Stopping class to use, defaults to None

        verbose: bool = True:
            Print the metrics logged while training or not

        binary_classifier_threshold: Optional[float] = None:
            Used in binar classification to log the metrics (F1 and Accuracy), if not set, will default to 0.5

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use when training
    """

    def __init__(
        self,
        task_type: Literal["regression", "binary", "multiclass"],
        num_epochs: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader[Any],
        test_dataloader: DataLoader[Any],
        num_classes: int,
        model_instance: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        model_uri: Optional[str] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        early_stopper: Optional[EarlyStopping] = None,
        verbose: bool = True,
        binary_decision_threshold: float = 0.5,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.binary_decision_threshold = binary_decision_threshold
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.early_stopper = early_stopper
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.task_type = task_type
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.verbose = verbose

        self._strategy = None

        self.model: torch.nn.Module
        if model_uri is not None:
            self.model = mlflow.pytorch.load_model(model_uri)  # type: ignore
        elif model_path is not None:
            self._load_model()
        elif model_instance is not None:
            self.model = model_instance
        else:
            raise ValueError(
                "Either `model_instance`, `model_path`, or `model_uri` should be passed, got None for all"
            )
        self._load_adaptor()

        if self.task_type == "regression":
            raise NotImplementedError(
                "Still didn't implement training regression models"
            )

    def _verify_model_path(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        return True

    def _load_model(self):
        if self._verify_model_path():
            self.model = torch.load(self.model_path, weights_only=True)
        else:
            raise InvalidModelPathError(
                "The model path is invalid, please verify that the path exists."
            )

    def save_as_torch(self, save_path: str, model_name: str, dummy_input: Any):
        assert self.model is not None, "No model to save"
        if not model_name.endswith(".pt"):
            model_name += ".pt"
        if len(save_path) == 0:
            raise ValueError("save_path should not be empty")

        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, model_name)
        else:
            raise ValueError("save_path must be a valid directory path")

        try:
            model: torch.jit.ScriptModule = torch.jit.script(self.model)  # type: ignore
        except Exception as e:
            logger.warning(
                f"Scripting Failed, falling back to tracing. Error details: {e}",
                extra={"error_details": e, "format": "pt"},
            )
            self.model.to(self.device)
            try:
                model = torch.jit.trace(self.model, dummy_input)  # type: ignore
            except Exception as e:
                logger.warning(
                    f"Failed to trace the model. Error details: {e}",
                    extra={"error_details": e, "format": "pt"},
                )
                return
        model.save(save_path)  # type: ignore
        logger.info(
            f"Model exported to TorchScript successfully and saved to {save_path}",
            extra={"save_path": save_path, "format": "pt"},
        )

    def save_as_onnx(
        self,
        save_path: str,
        model_name: str,
        dummy_input: Any,
        dynamic_shapes: dict[str, dict[int, str]],
        input_names: list[str] = ["input"],
        output_names: list[str] = ["output"],
    ):
        assert self.model is not None, "No model to save"
        if len(save_path) == 0:
            raise ValueError("save_path should not be empty")

        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, model_name)
        else:
            raise ValueError("save_path must be a valid directory path")

        model = self.model.cpu()
        dummy_input = dummy_input.to("cpu")
        if not model_name.endswith(".onnx"):
            model_name += ".onnx"
        try:
            torch.onnx.export(  # type: ignore
                model,
                dummy_input,
                save_path,
                dynamo=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
            )
            logger.info(
                f"Model exported to ONNX successfully and saved to {save_path}",
                extra={"save_path": save_path, "format": "onnx"},
            )
        except Exception as e:
            logger.warning(
                f"Failed to export model to ONNX. Error details: {e}",
                extra={"error_details": e, "format": "onnx"},
            )

    def _load_adaptor(self):
        assert self.model is not None, "Model is not loaded, cannot load strategy"
        if self.task_type == "binary":
            self._strategy = _BinaryClassifierTrainer(
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                model_instance=self.model,
                binary_decision_threshold=self.binary_decision_threshold,
                device=self.device,
            )
        elif self.task_type == "multiclass":
            self._strategy = _MulticlassClassifierTrainer(
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                model_instance=self.model,
                num_classes=self.num_classes,
                device=self.device,
            )
        elif self.task_type == "regression":
            raise NotImplementedError("Regression task is not implemented yet")
        else:
            raise ValueError(
                "Invalid task type, supported values are: 'binary', 'multiclass', 'regression'"
            )

    def _train_loop(self) -> tuple[float, dict[str, float]]:
        assert self.model is not None, (
            "Model is not loaded, cannot perform training loop"
        )
        assert self._strategy is not None, (
            "Strategy is not loaded, cannot perform training loop"
        )
        self.model.train()
        train_loss = 0.0
        for X_batch, y_batch in self.train_dataloader:
            X_batch, y_batch = (
                X_batch.to(self.device, non_blocking=True),
                y_batch.to(self.device, non_blocking=True),
            )  # non_blocking works when the dataloader is set with pin_memory = True so make sure to do that for effeciency

            batch_loss, batch_preds = self._strategy.step(X_batch, y_batch, train=True)

            self._strategy.metrics.update(batch_preds, y_batch)
            train_loss += batch_loss

        if self.lr_scheduler:
            self.lr_scheduler.step()

        train_metrics = self._strategy.metrics.compute()
        self._strategy.metrics.reset()

        train_loss /= len(self.train_dataloader)
        return train_loss, train_metrics

    def _test_loop(self) -> tuple[float, dict[str, float]]:
        assert self.model is not None, (
            "Model is not loaded, cannot perform testing loop"
        )
        assert self._strategy is not None, (
            "Strategy is not loaded, cannot perform testing loop"
        )
        assert self.model is not None, (
            "Model is not loaded, cannot perform testing loop"
        )
        test_loss = 0.0
        self.model.eval()
        for X_batch, y_batch in self.test_dataloader:
            X_batch, y_batch = (
                X_batch.to(self.device, non_blocking=True),
                y_batch.to(self.device, non_blocking=True),
            )

            batch_loss, batch_preds = self._strategy.step(X_batch, y_batch, train=False)

            self._strategy.metrics.update(batch_preds, y_batch)
            test_loss += batch_loss

        test_metrics = self._strategy.metrics.compute()
        self._strategy.metrics.reset()

        test_loss /= len(self.test_dataloader)
        return test_loss, test_metrics

    def train(self, log_every: int = 10) -> tuple[float, float]:
        """
        Trains the provided model

        ---
        Returns:
            train_loss: float:
                Training Loss
            test_loss: float:
                Testing Loss
        """
        assert self.model is not None, "Model is not loaded, cannot perform training"
        self.model.to(self.device, non_blocking=True)
        train_loss, test_loss = 0.0, 0.0
        train_metrics, test_metrics = {}, {}
        for epoch in tqdm(range(self.num_epochs)):
            train_loss, train_metrics = self._train_loop()
            test_loss, test_metrics = self._test_loop()

            if self.verbose:
                if epoch % log_every == 0:
                    train_metrics_str = " | ".join(
                        f"{' '.join(k.title().split('_'))}: {v:.2%}"
                        for k, v in train_metrics.items()
                    )
                    test_metrics_str = " | ".join(
                        f"{' '.join(k.title().split('_'))}: {v:.2%}"
                        for k, v in test_metrics.items()
                    )
                    logger.info(
                        f""" Epoch {epoch}
                        Training Loss = {train_loss:.2f}\t| Testing Loss = {test_loss:.2f}
                        Training Metrics:
                            {train_metrics_str}
                        Testing Metrics:
                            {test_metrics_str}
                        __________________________________________________________________________________
                        """
                    )
            if self.early_stopper:
                self.early_stopper(test_loss, self.model)
                if self.early_stopper.early_stop:
                    logger.info(
                        "================================ Early Stopping ================================"
                    )
                    self.early_stopper.load_best_model(self.model)
                    break

        train_metrics_str = " | ".join(
            f"{' '.join(k.title().split('_'))}: {v:.2%}"
            for k, v in train_metrics.items()
        )
        test_metrics_str = " | ".join(
            f"{' '.join(k.title().split('_'))}: {v:.2%}"
            for k, v in test_metrics.items()
        )
        logger.info(
            f""" Final Results:\nTraining Loss = {train_loss:.2f}\t| Testing Loss = {test_loss:.2f}\nTraining Metrics:\n    {train_metrics_str}\t| Testing Metrics\n    {test_metrics_str}
            """
        )
        return train_loss, test_loss


class _TrainingStrategy:
    """
    This class acts as an abstraction layer for a training strategy, it defines the step function that performs a training step and returns the loss and the predictions to log the metrics with.

    Args:
        loss_fn: torch.nn.Module:
            The loss function to use

        optimizer: torch.optim.Optimizer:
            The optimizer to use

        model_instance: torch.nn.Module:
            The model to train

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use when training
    """

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_instance: torch.nn.Module,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model_instance

    def step(
        self, X: torch.Tensor, y: torch.Tensor, train: bool
    ) -> tuple[float, torch.Tensor]:
        """Performs a training step, returns (loss, processed_predictions)"""
        raise NotImplementedError("This method should be implemented by a subclass")

    def _calculate_step(self, X: torch.Tensor, y: torch.Tensor, train: bool):
        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            logits = self.model(X)
            loss = self.loss_fn(input=logits, target=y)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        return loss, logits


class _BinaryClassifierTrainer(_TrainingStrategy):
    """
    Trains a binary classifier

    ---
    Args:
        loss_fn: torch.nn.modules.loss:
            The loss function to use

        optimizer: torch.optim:
            The optimizer to use

        model_instance: torch.nn.Module:
            The model to train

        binary_classifier_threshold: Optional[float] = None:
            Used in binar classification to log the metrics (F1 and Accuracy), if not set, will default to 0.5

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use when training
    """

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_instance: torch.nn.Module,
        binary_decision_threshold: float = 0.5,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            model_instance=model_instance,
        )
        self.device = torch.device(device)
        self.binary_decision_threshold = binary_decision_threshold
        self.metrics = set_classification_metrics(
            num_classes=2, task="binary", device=self.device
        )

    def step(
        self, X: torch.Tensor, y: torch.Tensor, train: bool
    ) -> tuple[float, torch.Tensor]:
        """Training or Training step for a binary classifier, returns (loss, processed_predictions)"""
        assert self.model is not None, (
            "Model is not loaded, cannot perform training step"
        )

        loss, logits = self._calculate_step(X, y, train)

        probs = torch.sigmoid(logits)
        preds = probs > self.binary_decision_threshold
        return loss.item(), preds


class _MulticlassClassifierTrainer(_TrainingStrategy):
    """
    Trains a multiclass classifier

    ---
    Args:
        loss_fn: torch.nn.modules.loss:
            The loss function to use

        optimizer: torch.optim:
            The optimizer to use

        model_instance: torch.nn.Module:
            The model to train

        num_classes: int:
            The number of classes in the classification task

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use when training
    """

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_instance: torch.nn.Module,
        num_classes: int,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            model_instance=model_instance,
        )
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.metrics = set_classification_metrics(
            num_classes=self.num_classes, task="multiclass", device=self.device
        )

    def step(
        self, X: torch.Tensor, y: torch.Tensor, train: bool
    ) -> tuple[float, torch.Tensor]:
        """Training or Training step for a binary classifier, returns (loss, processed_predictions)"""
        assert self.model is not None, (
            "Model is not loaded, cannot perform training step"
        )

        loss, logits = self._calculate_step(X, y, train)

        probs = torch.softmax(input=logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return loss.item(), preds
