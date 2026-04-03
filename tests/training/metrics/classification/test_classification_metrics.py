from ml_project_template.training.metrics.classification.metrics import (  # type: ignore
    classification_report,
    set_classification_metrics,
)
import torch


def test_set_classification_metrics():
    multiclass_metrics = set_classification_metrics(
        task="multiclass",
        num_classes=3,
    )

    binary_metrics = set_classification_metrics(task="binary")

    assert "accuracy" in multiclass_metrics
    assert "precision" in multiclass_metrics
    assert "recall" in multiclass_metrics
    assert "f1_score" in multiclass_metrics

    assert "accuracy" in binary_metrics
    assert "precision" in binary_metrics
    assert "recall" in binary_metrics
    assert "f1_score" in binary_metrics


def test_classification_report():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 2, 1, 0, 0, 1])

    multiclass_report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        task="multiclass",
        num_classes=3,
    )

    y_true = torch.tensor([0, 1, 1, 0, 1, 0])
    y_pred = torch.tensor([0, 0, 1, 0, 0, 1])

    binary_report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        task="binary",
    )

    assert "accuracy" in multiclass_report
    assert "precision" in multiclass_report
    assert "recall" in multiclass_report
    assert "f1_score" in multiclass_report

    assert "accuracy" in binary_report
    assert "precision" in binary_report
    assert "recall" in binary_report
    assert "f1_score" in binary_report
