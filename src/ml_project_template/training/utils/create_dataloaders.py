from torch.utils.data import DataLoader
from ..architecture.dataset import MockDataset
from typing import Any


def create_data_loader(
    features: Any,
    labels: Any,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader[Any]:
    """Creates `DataLoader` object using the project's `Dataset` class

    Args:
        features: Any:
            The features to include in the dataset.

        labels: Any:
            The labels to include in the dataset.

        batch_size: int = 64:
            The batch size.

        shuffle: bool = False:
            Whether to shuffle the data in the `DataLoader` or not.

    Returns:
        DataLoader[Any]:
            The `DataLoader` object built from the dataset
    """
    dataset = MockDataset(features=features, labels=labels)
    pin_memory = True
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )  # type: ignore
