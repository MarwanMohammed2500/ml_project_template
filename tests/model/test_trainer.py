import torch
import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset
from ml_project_template.model.trainer import Trainer, _BinaryClassifierTrainer  # type: ignore
from typing import Any


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def trainer_params() -> dict[str, Any]:
    """Provides standard params for initializing a Trainer in tests"""
    X = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,)).float().unsqueeze(1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=2)

    model = SimpleModel()
    return {
        "task_type": "binary",
        "num_epochs": 2,
        "loss_fn": torch.nn.BCEWithLogitsLoss(),
        "optimizer": torch.optim.Adam(model.parameters(), lr=0.01),
        "train_dataloader": dl,
        "test_dataloader": dl,
        "num_classes": 2,
        "model_instance": model,
        "device": "cpu",
    }


def test_trainer_raises_error_if_no_model_provided(trainer_params: dict[str, Any]):
    trainer_params["model_instance"] = None
    trainer_params["model_path"] = None
    trainer_params["model_uri"] = None

    with pytest.raises(
        ValueError,
        match="Either `model_instance`, `model_path`, or `model_uri` should be passed, got None for all",
    ):
        Trainer(**trainer_params)


def test_trainer_loads_correct_strategy(trainer_params: dict[str, Any]):
    trainer: Trainer = Trainer(**trainer_params)

    assert isinstance(trainer._strategy, _BinaryClassifierTrainer)  # type: ignore


def test_train_loop_updates_weights(trainer_params: dict[str, Any]):
    trainer: Trainer = Trainer(**trainer_params)

    # Capture weights before training
    assert trainer.model is not None
    initial_weights = trainer.model.fc.weight.clone()  # type: ignore

    trainer._train_loop()  # type: ignore

    assert not torch.equal(initial_weights, trainer.model.fc.weight)  # type: ignore


def test_test_loop_does_not_update_weights(trainer_params: dict[str, Any]):
    trainer: Trainer = Trainer(**trainer_params)

    # Capture weights before testing
    initial_weights = trainer.model.fc.weight.clone()  # type: ignore

    trainer._test_loop()  # type: ignore

    assert torch.equal(initial_weights, trainer.model.fc.weight)  # type: ignore


@patch("torch.onnx.export")
@patch("os.makedirs")
def test_save_as_onnx_calls_torch_export(
    mock_makedirs: MagicMock, mock_export: MagicMock, trainer_params: dict[str, Any]
):
    trainer: Trainer = Trainer(**trainer_params)
    dummy_input = torch.randn(1, 10)

    trainer.save_as_onnx(
        save_path="/tmp/test_dir",
        model_name="test_model",
        dummy_input=dummy_input,
        dynamic_shapes={"input": {0: "batch"}},
    )

    # Verify directory was created and export was called
    mock_makedirs.assert_called_once()
    mock_export.assert_called_once()
