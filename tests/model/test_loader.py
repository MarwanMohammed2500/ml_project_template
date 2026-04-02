from unittest.mock import MagicMock, patch
from ml_project_template.model import Model # type: ignore

@patch("src.ml_project_template.model.ort.InferenceSession")
def test_binary_model_preload_initializes_strategy(mock_session): # type: ignore
    mock_session.return_value = MagicMock()
    
    model = Model(
        model_path="fake/path.onnx",
        task_type="binary",
        decision_threshold=0.7
    )
    
    with patch.object(Model, "_verify_model_path", return_value=True):
        model.preload()
    
    assert model.loaded is True
    assert model._strategy is not None # type: ignore
    
    from src.ml_project_template.model.loader import _BinaryClassifierModel # type: ignore
    assert isinstance(model._strategy, _BinaryClassifierModel) # type: ignore

@patch("src.ml_project_template.model.ort.InferenceSession")
def test_multiclass_model_preload_initializes_strategy(mock_session): # type: ignore
    mock_session.return_value = MagicMock()
    
    model = Model(
        model_path="fake/path.onnx",
        task_type="multiclass",
    )
    
    with patch.object(Model, "_verify_model_path", return_value=True):
        model.preload()
    
    assert model.loaded is True
    assert model._strategy is not None # type: ignore

    from src.ml_project_template.model.loader import _MulticlassClassifierModel # type: ignore
    assert isinstance(model._strategy, _MulticlassClassifierModel) # type: ignore

import pytest

@patch("src.ml_project_template.model.ort.InferenceSession")
def test_regression_raises_not_implemented(mock_session): # type: ignore
    mock_session.return_value = MagicMock()
    model = Model(model_path="fake.onnx", task_type="regression")
    
    with patch.object(Model, "_verify_model_path", return_value=True):
        with pytest.raises(NotImplementedError, match="Regression task is not implemented"):
            model.preload()

## Implement tests for predict