from ml_project_template.metrics.regression.metrics import ( # type: ignore
    regression_report,
    set_regression_metrics,
)
import torch
def test_set_regression_metrics():
    metrics = set_regression_metrics(
        num_outputs = 1,
        nrmse_normalization_technique = "mean",
        mse_squared = True,
    )
    
    assert "normalized_root_mean_squared_error" in metrics
    assert "mean_squared_error" in metrics

def test_regression_report():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 2, 1, 0, 0, 1])
    
    report = regression_report(
        y_true=y_true,
        y_pred=y_pred,
        nrmse_normalization_technique = "mean",
        mse_squared=True,
    )
    
    assert "normalized_root_mean_squared_error" in report
    assert "mean_squared_error" in report