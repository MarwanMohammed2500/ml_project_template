from ml_project_template.services.inference import load_model, predict  # type: ignore
from unittest.mock import patch, MagicMock


@patch("src.ml_project_template.model.ort.InferenceSession")
def test_predict(mock_session):  # type: ignore
    mock_session.return_value = MagicMock()

    # Load the model first
    load_model()

    # Call predict with some input and verify it returns expected output
    input_data = [1.0, 2.0, 3.0]
    output, prob = predict(input_data)
    assert output is not None
    assert prob is not None
    assert isinstance(prob, float)
    assert isinstance(
        output, int
    )  # Assuming classification for this test, adjust as needed
