from unittest.mock import MagicMock, patch
from ml_project_template.serving.model import Model  # type: ignore
from ml_project_template.core.errors.exceptions import InvalidModelPathError  # type: ignore
import pytest
import numpy as np


@patch("onnxruntime.InferenceSession")
def test_binary_model_preload_initializes_strategy(mock_session):  # type: ignore
    mock_session.return_value = MagicMock()

    model = Model(
        model_uri="models:/SimpleModel_ONNX@production",
        task_type="binary",
        decision_threshold=0.7,
    )

    with patch.object(Model, "_verify_model_uri", return_value=True):
        model.preload()

    assert model.loaded is True
    assert model._strategy is not None  # type: ignore

    from ml_project_template.serving.model.loader import _BinaryClassifierModel  # type: ignore

    assert isinstance(model._strategy, _BinaryClassifierModel)  # type: ignore


@patch("onnxruntime.InferenceSession")
def test_multiclass_model_preload_initializes_strategy(mock_session):  # type: ignore
    mock_session.return_value = MagicMock()

    model = Model(
        model_uri="models:/SimpleModel_ONNX@production",
        task_type="multiclass",
    )

    with patch.object(Model, "_verify_model_uri", return_value=True):
        model.preload()

    assert model.loaded is True
    assert model._strategy is not None  # type: ignore

    from ml_project_template.serving.model.loader import _MulticlassClassifierModel  # type: ignore

    assert isinstance(model._strategy, _MulticlassClassifierModel)  # type: ignore


@patch("onnxruntime.InferenceSession")
def test_regression_raises_not_implemented(mock_session):  # type: ignore
    mock_session.return_value = MagicMock()
    model = Model(
        model_uri="models:/SimpleModel_ONNX@production", task_type="regression"
    )

    with patch.object(Model, "_verify_model_uri", return_value=True):
        with pytest.raises(
            NotImplementedError, match="Regression task is not implemented"
        ):
            model.preload()


@patch("onnxruntime.InferenceSession")
def test_predict(mock_session):  # type: ignore
    mock_run = MagicMock()
    mock_run.return_value = [np.array([0.7], dtype=np.float32)]
    mock_session.return_value.run = mock_run  # type: ignore
    model = Model(model_uri="models:/SimpleModel_ONNX@production", task_type="binary")

    with patch.object(Model, "_verify_model_uri", return_value=True):
        model.preload()

    input_data = [1.0, 2.0, 3.0]
    output, prob = model.predict(input=input_data)

    assert output is not None
    assert prob is not None
    assert isinstance(output, int)
    assert isinstance(prob, float)


def test_invalid_task_type_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Invalid task type, supported values are: 'binary', 'multiclass', 'regression'",
    ):
        Model(model_uri="models:/SimpleModel_ONNX@production", task_type="invalid_task")  # type: ignore - it is supposed to be wrong


def test_invalid_model_uri_raises_value_error():
    model = Model(model_uri="invalid/path.onnx", task_type="binary")

    with pytest.raises(
        InvalidModelPathError,
        match="The model path is invalid, please verify if the model has the correct extention, or that the path exists.",
    ):
        model.preload()


def test_model_uri_with_invalid_extension_raises_value_error():
    model = Model(model_uri="model.txt", task_type="binary")

    with pytest.raises(
        InvalidModelPathError,
        match="The model path is invalid, please verify if the model has the correct extention, or that the path exists.",
    ):
        model.preload()
