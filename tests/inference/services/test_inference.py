from unittest.mock import patch, MagicMock
from ml_project_template.serving.services import inference  # type: ignore

@patch("ml_project_template.serving.services.inference.download_artifacts")
@patch("ml_project_template.serving.services.inference.Model")
def test_predict(mock_model_cls, mock_download):  # type: ignore
    # Create a mock model instance
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = (1, 0.9)
    mock_model_cls.return_value = mock_model_instance
    mock_download.return_value = "/tmp/fake_model_path"

    inference.load_model()

    input_data = [1.0, 2.0, 3.0]
    output, prob = inference.predict(input_data)

    assert output is not None
    assert prob is not None
    assert isinstance(output, str)
    assert isinstance(prob, float)
