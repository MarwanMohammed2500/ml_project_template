from torch.utils.data import DataLoader
from .dataset import MockDataset
from typing import Any


def create_data_loader(
    features: Any,
    labels: Any,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader[Any]:
    dataset = MockDataset(features=features, labels=labels)
    pin_memory = True
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )  # type: ignore
