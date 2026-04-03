import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from ml_project_template.training.utils import create_data_loader  # type: ignore
from typing import Any


class MockDataset(Dataset):  # type: ignore
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


@pytest.fixture
def sample_data() -> dict[str, Any]:
    features = torch.arange(20).view(10, 2)
    labels = torch.arange(10)
    dataset_inputs = {"features": features, "labels": labels}
    return dataset_inputs


def test_returns_dataloader(sample_data: dict[str, Any]):
    dataset_inputs = sample_data
    loader = create_data_loader(
        dataset_class=MockDataset, dataset_inputs=dataset_inputs
    )
    assert isinstance(loader, DataLoader)


def test_batch_size_respected(sample_data: dict[str, Any]):
    dataset_inputs = sample_data
    loader = create_data_loader(
        batch_size=4, dataset_class=MockDataset, dataset_inputs=dataset_inputs
    )
    for batch_features, batch_labels in loader:
        assert batch_features.shape[0] <= 4
        assert batch_labels.shape[0] <= 4


def test_shuffle_behavior(sample_data: dict[str, Any]):
    dataset_inputs = sample_data
    loader_no_shuffle = create_data_loader(
        shuffle=False, dataset_class=MockDataset, dataset_inputs=dataset_inputs
    )
    loader_shuffle = create_data_loader(
        shuffle=True, dataset_class=MockDataset, dataset_inputs=dataset_inputs
    )

    batch_features, batch_labels = next(iter(loader_no_shuffle))
    assert torch.equal(
        batch_features.flatten(),
        dataset_inputs["features"][: loader_no_shuffle.batch_size].flatten(),
    )
    assert torch.equal(
        batch_labels.flatten(),
        dataset_inputs["labels"][: loader_no_shuffle.batch_size].flatten(),
    )

    batch_features_sh, batch_labels_sh = next(iter(loader_shuffle))

    assert set(batch_features_sh.flatten().tolist()) <= set(
        dataset_inputs["features"].flatten().tolist()
    )  # type: ignore
    assert set(batch_labels_sh.flatten().tolist()) <= set(
        dataset_inputs["labels"].tolist()
    )  # type: ignore


def test_data_integrity(sample_data: dict[str, Any]):
    dataset_inputs = sample_data
    loader = create_data_loader(
        batch_size=2, dataset_class=MockDataset, dataset_inputs=dataset_inputs
    )

    all_features = torch.cat([batch[0] for batch in loader], dim=0)
    all_labels = torch.cat([batch[1] for batch in loader], dim=0)

    assert torch.equal(all_features.flatten(), dataset_inputs["features"].flatten())
    assert torch.equal(all_labels.flatten(), dataset_inputs["labels"].flatten())
