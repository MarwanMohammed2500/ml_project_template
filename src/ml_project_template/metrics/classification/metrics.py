import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from typing import Literal, Optional


def set_classification_metrics(
    task: Literal["multiclass", "binary"],
    num_classes: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> MetricCollection:
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
        accuracy_score = Accuracy(task=task)
        f1_score = F1Score(task=task)
        precision = Precision(task=task)
        recall = Recall(task=task)
    elif task == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes must be set if task is not binary")
        accuracy_score = Accuracy(task=task, num_classes=num_classes)
        f1_score = F1Score(task=task, num_classes=num_classes)
        precision = Precision(task=task, num_classes=num_classes)
        recall = Recall(task=task, num_classes=num_classes)
    else:
        raise ValueError(
            f"`task` should be `binary` or `multiclass` got {task} instead"
        )
    return MetricCollection(
        {
            "accuracy": accuracy_score,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
        }
    ).to(device)


def classification_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    task: Literal["multiclass", "binary"],
    num_classes: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict[str, float]:
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
    metrics = set_classification_metrics(
        num_classes=num_classes, task=task, device=device
    )

    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    report = metrics(y_pred, y_true)
    return report
