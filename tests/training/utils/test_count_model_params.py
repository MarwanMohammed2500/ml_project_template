import torch
import pandas as pd
import pytest
from torch import nn
from ml_project_template.training.utils import count_model_parameters  # type: ignore


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def model_and_input() -> tuple[nn.Module, torch.Tensor]:
    model = SimpleModel()
    x = torch.randn(1, 4)
    return model, x


def test_returns_dataframe(model_and_input: tuple[nn.Module, torch.Tensor]):
    model, x = model_and_input
    df = count_model_parameters(model, x)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_dataframe_columns(model_and_input: tuple[nn.Module, torch.Tensor]):
    model, x = model_and_input
    df = count_model_parameters(model, x)

    expected_cols = {"layer", "input_shape", "output_shape", "num_params"}
    assert expected_cols.issubset(df.columns)


def test_layer_entries_exist(model_and_input: tuple[nn.Module, torch.Tensor]):
    model, x = model_and_input
    df = count_model_parameters(model, x)

    layers = df["layer"].tolist()
    assert "Linear" in layers
    assert "ReLU" in layers


def test_parameter_counts_correct(model_and_input: tuple[nn.Module, torch.Tensor]):
    model, x = model_and_input
    df = count_model_parameters(model, x)

    fc1_params = sum(p.numel() for p in model.fc1.parameters())  # type: ignore
    fc2_params = sum(p.numel() for p in model.fc2.parameters())  # type: ignore

    linear_rows = df[df["layer"] == "Linear"]

    assert fc1_params in linear_rows["num_params"].values
    assert fc2_params in linear_rows["num_params"].values


def test_shapes_are_lists(model_and_input: tuple[nn.Module, torch.Tensor]):
    model, x = model_and_input
    df = count_model_parameters(model, x)

    for _, row in df.iterrows():
        assert isinstance(row["input_shape"], list)
        assert isinstance(row["output_shape"], list)
