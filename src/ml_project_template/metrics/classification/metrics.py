import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall
from typing import Literal, Optional


def set_classification_metrics(
    task: Literal["multiclass", "binary"],
    num_classes: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Creates TorchMetrics instances of accuracy, precision, recall, and f1.

    Args:
        task: Literal["multiclass", "binary"]:
            The classification task

        num_classes: Optional[int] = None:
            Number of classes, must be set if task == multiclass

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use
    """
    if task == "binary":
        accuracy_score = Accuracy(task=task).to(device)
        f1_score = F1Score(task=task).to(device)
        precision = Precision(task=task).to(device)
        recall = Recall(task=task).to(device)
    elif task == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes must be set if task is not binary")
        accuracy_score = Accuracy(task=task, num_classes=num_classes).to(device)
        f1_score = F1Score(task=task, num_classes=num_classes).to(device)
        precision = Precision(task=task, num_classes=num_classes).to(device)
        recall = Recall(task=task, num_classes=num_classes).to(device)
    else:
        raise ValueError(
            f"`task` should be `binary` or `multiclass` got {task} instead"
        )
    return accuracy_score, f1_score, precision, recall


def classification_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    task: Literal["multiclass", "binary"],
    num_classes: Optional[int] = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Creates a classification report using accuracy, precision, recall, and f1-score

    Args:
        y_true: torch.Tensor:
            The targets

        y_pred: torch.Tensor:
            The model's predictions

        task: Literal["multiclass", "binary"]:
            The classification task

        num_classes: Optional[int] = None:
            Number of classes, must be set if task == multiclass

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use
    """
    accuracy_score, f1_score, precision, recall = set_classification_metrics(
        num_classes=num_classes, task=task, device=device
    )
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true).to(device)

    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred).to(device)

    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    report = {
        "Accuracy": accuracy_score(y_pred, y_true).item(),
        "Precision": precision(y_pred, y_true).item(),
        "Recall": recall(y_pred, y_true).item(),
        "F1-Score": f1_score(y_pred, y_true).item(),
    }
    return report
