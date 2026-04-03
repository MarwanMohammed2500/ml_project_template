from torch.utils.data import Dataset
import numpy.typing as npt
import numpy as np
import torch


class MockDataset(Dataset):  # type: ignore
    def __init__(
        self, features: npt.NDArray[np.float32], labels: npt.NDArray[np.float32]
    ):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            [self.labels[idx]], dtype=torch.long
        ).float()
