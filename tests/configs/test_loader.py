from ml_project_template.configs import load_configs # type: ignore

def test_load_configs():
    configs = load_configs("tests/configs/test_configs.yaml")
    assert configs is not None
    assert configs.model_config is not None
    
    # Assert all configurations you have in your test_configs.yaml file
    # assert configs.training_config is not None
    # assert configs.data_config is not None
    # assert configs.model_config.pretrained_model_path == "path/to/pretrained/model"
    # assert configs.training_config.binary_decision_threshold == 0.5
    # assert configs.data_config.train_data_path == "path/to/train/data"