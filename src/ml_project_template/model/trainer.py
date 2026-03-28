from src.ml_project_template.errors import InvalidModelPathError
from src.ml_project_template.metrics import (
    set_classification_metrics,
)  # add regression related ones when implementing regression training.
from typing import Literal, Optional, Any
from tqdm.auto import tqdm
import logging
import torch
import os

logger = logging.getLogger(__name__)


class SupervisedModelTrainer:
    """
    Trains a supervised model

    ---
    Args:
        task_type:Literal["regression", "binary", "multiclass"]:
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
        loss_fn: torch.nn.modules.loss,
        optimizer: torch.optim,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        num_classes: int,
        model: Optional[torch.nn.Module] = None,
        pretrained_model_path: Optional[str] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler] = None,
        early_stopper: Optional[type] = None,
        verbose: bool = True,
        binary_decision_threshold: Optional[float] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):

        self.binary_decision_threshold = binary_decision_threshold
        self.pretrained_model_path = pretrained_model_path
        self.device = torch.device(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.early_stopper = early_stopper
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.task_type = task_type
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.model = model

        if self.task_type != "regression":
            self.accuracy_score, self.f1_score, _, _ = set_classification_metrics(
                num_classes=self.num_classes, task=self.task_type, device=self.device
            )

            if self.task_type == "binary":
                if self.binary_decision_threshold is None:
                    self.binary_decision_threshold = 0.5
        if self.task_type == "regression":
            raise NotImplementedError(
                "Still didn't implement training regression models"
            )

        if not self.model and not self.pretrained_model_path:
            raise ValueError(
                "Either `model` should be passed or `pretrained_model_path` should be passed, go None for both"
            )

        if self.pretrained_model_path:
            self._load_model()

    def _training_step(self, X, y):
        logits = self.model(X)
        loss = self.loss_fn(
            input=logits, target=y
        )  # `target` here can change depending on the loss function and model (casted to long, squeezed/unsqueezed on a certain dimension, etc.) modify according to your needs
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item(), logits

    def _testing_step(self, X, y):
        with torch.inference_mode():
            logits = self.model(X)
            loss = self.loss_fn(input=logits, target=y)
        return loss.item(), logits

    def _train_loop(self):
        self.model.train()
        train_loss = 0.0
        for X_batch, y_batch in self.train_dataloader:
            X_batch, y_batch = (
                X_batch.to(self.device, non_blocking=True),
                y_batch.to(self.device, non_blocking=True),
            )  # non_blocking works when the dataloader is set with pin_memory = True so make sure to do that for effeciency

            batch_loss, batch_logits = self._training_step(X_batch, y_batch)

            if self.task_type == "binary":
                probs = torch.sigmoid(batch_logits)
                preds = probs > self.binary_classifier_threshold
            elif self.task_type == "multiclass":
                probs = torch.softmax(input=batch_logits, dim=-1)
                preds = probs.argmax(dim=-1)

            self.accuracy_score.update(preds, y_batch)
            self.f1_score.update(preds, y_batch)
            train_loss += batch_loss

        if self.lr_scheduler:
            self.lr_scheduler.step()

        acc = self.accuracy_score.compute()
        f1 = self.f1_score.compute()
        self.accuracy_score.reset()
        self.f1_score.reset()

        train_loss /= len(self.train_dataloader)
        return train_loss, acc, f1

    def _test_loop(self):
        test_loss = 0.0
        self.model.eval()
        for X_batch, y_batch in self.test_dataloader:
            X_batch, y_batch = (
                X_batch.to(self.device, non_blocking=True),
                y_batch.to(self.device, non_blocking=True),
            )

            batch_loss, batch_logits = self._testing_step(X_batch, y_batch)

            if self.task_type == "binary":
                probs = torch.sigmoid(batch_logits)
                preds = probs > self.binary_classifier_threshold
            elif self.task_type == "multiclass":
                probs = torch.softmax(input=batch_logits, dim=-1)
                preds = probs.argmax(dim=-1)

            self.accuracy_score.update(preds, y_batch)
            self.f1_score.update(preds, y_batch)
            test_loss += batch_loss

        acc = self.accuracy_score.compute()
        f1 = self.f1_score.compute()
        self.accuracy_score.reset()
        self.f1_score.reset()

        test_loss /= len(self.test_dataloader)
        return test_loss, acc, f1

    def train(self):
        """
        Trains the provided model

        ---
        Returns:
            train_loss: float:
                Training Loss
            test_loss: float:
                Testing Loss
        """
        self.model.to(self.device, non_blocking=True)
        for epoch in tqdm(range(self.num_epochs)):
            train_loss, train_accuracy_score, train_f1_score = self._train_loop()
            test_loss, test_accuracy_score, test_f1_score = self._test_loop()

            if epoch % 10 == 0:
                if self.verbose:
                    logger.info(
                        f""" Epoch {epoch}
                        Training Loss = {train_loss:.2f}\t| Testing Loss = {test_loss:.2f}
                        Training Accuracy = {train_accuracy_score:.2%}\t| Testing Accuracy = {test_accuracy_score:.2%}
                        Training F1-Score = {train_f1_score:.2%}\t| Testing F1-Score = {test_f1_score:.2%}
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

        logger.info(
            f""" Final Results:
            Training Loss = {train_loss:.2f}\t| Testing Loss = {test_loss:.2f}
            Training Accuracy = {train_accuracy_score:.2%}\t| Testing Accuracy = {test_accuracy_score:.2%}
            Training F1-Score = {train_f1_score:.2%}\t| Testing F1-Score = {test_f1_score:.2%}
            """
        )
        return train_loss, test_loss

    def _verify_model_path(self) -> bool:
        if not os.path.exists(self.pretrained_model_path):
            return False
        return True

    def _load_model(self):
        if self._verify_model_path():
            self.model = torch.load(self.pretrained_model_path, weights_only=True)
        else:
            raise InvalidModelPathError(
                "The model path is invalid, please verify that the path exists."
            )

    def save_as_torch(self, save_path: str, model_name: str, dummy_input: Any):
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
            model = torch.jit.script(self.model)
        except Exception as e:
            logger.warning(
                f"Scripting Failed, falling back to tracing. Error details: {e}",
                extra={"error_details": e, "format": "pt"},
            )
            self.model.to(self.device)
            try:
                model = torch.jit.trace(self.model, dummy_input)
            except Exception as e:
                logger.warning(
                    f"Failed to trace the model. Error details: {e}",
                    extra={"error_details": e, "format": "pt"},
                )
                return
        model.save(save_path)
        logger.info(
            f"Model exported to TorchScript successfully and saved to {save_path}",
            extra={"save_path": save_path, "format": "pt"},
        )

    def save_as_onnx(self, save_path: str, model_name: str, dummy_input: Any):
        if len(save_path) == 0:
            raise ValueError("save_path should not be empty")

        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, model_name)
        else:
            raise ValueError("save_path must be a valid directory path")

        model = self.model.to("cpu")
        dummy_input = dummy_input.to("cpu")
        if not model_name.endswith(".onnx"):
            model_name += ".onnx"
        try:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                dynamo=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_shapes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
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
