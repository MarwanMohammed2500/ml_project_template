from torchmetrics import MeanSquaredError, NormalizedRootMeanSquaredError
from typing import Literal
import torch


def set_regression_metrics(
    num_outputs: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    nrmse_normalization_technique: Literal["mean", "range", "std", "l2"] = "mean",
    mse_squared: bool = True,
):
    """
    Creates a Torchmetrics instances of Normalized Root Mean Squared Error (NRMSE) and Mean Squared Error (MSE)

    Args:
        num_outputs: int = 1:
            Number of outputs in multioutput setting

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use

        nrmse_normalization_technique: Literal['mean', 'range', 'std', 'l2'] = "mean":
            type of normalization to be applied to NRMSE.
            Choose from “mean”, “range”, “std”, “l2” which corresponds to normalizing the RMSE by the mean of the target, the range of the target,
            the standard deviation of the target or the L2 norm of the target.

        mse_squared:bool=True:
            If True returns MSE value, if False returns RMSE value.
    """
    nrmse = NormalizedRootMeanSquaredError(
        normalization=nrmse_normalization_technique,
        num_outputs=num_outputs,
        device=device,
    )
    mse = MeanSquaredError(squared=mse_squared, num_outputs=num_outputs, device=device)
    return nrmse, mse


def regression_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_outputs: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    nrmse_normalization_technique: Literal["mean", "range", "std", "l2"] = "mean",
    mse_squared: bool = True,
):
    """
    Creates a regression report using Normalized Root Mean Squared Error (NRMSE) and Mean Squared Error (MSE)

    Args:
        y_true: torch.Tensor:
            The targets

        y_pred: torch.Tensor:
            The model's predictions

        num_outputs: int = 1:
            Number of outputs in multioutput setting

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            The device to use

        nrmse_normalization_technique: Literal['mean', 'range', 'std', 'l2'] = "mean":
            type of normalization to be applied to NRMSE.
            Choose from “mean”, “range”, “std”, “l2” which corresponds to normalizing the RMSE by the mean of the target, the range of the target,
            the standard deviation of the target or the L2 norm of the target.

        mse_squared:bool=True:
            If True returns MSE value, if False returns RMSE value.
    """
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true).to(device)

    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred).to(device)

    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    nrmse, mse = set_regression_metrics(
        num_outputs=num_outputs,
        device=device,
        nrmse_normalization_technique=nrmse_normalization_technique,
        mse_squared=mse_squared,
    )

    return {
        "Normalized Root Mean Squared Error": nrmse(y_pred, y_true).item(),
        "Mean Squared Error" if mse_squared else "Root Mean Squared Error": mse(
            y_pred, y_true
        ).item(),
    }
