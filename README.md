# ML Project Template

A modular template for building **machine learning training pipelines and inference APIs** with PyTorch and FastAPI.

This template provides a structured starting point for ML projects, including:

- Model training utilities
- Evaluation metrics
- Early stopping
- Configuration validation
- FastAPI inference endpoints
- Schema-based request/response validation
- Logging setup
- Structured project architecture

The goal is to provide a **clean foundation for building production-ready ML services**.

---

# Features

- **PyTorch Training Framework**
  - Modular trainer class
  - Device-aware training
  - Support for binary and multiclass classification
  - Extendable to regression

- **Metrics System**
  - Built with `torchmetrics`
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Regression metrics

- **Early Stopping**
  - Patience-based training stopping
  - Automatic best model restoration

- **FastAPI Inference API**
  - `/api/predict` endpoint
  - `/health` health check
  - Pydantic request and response schemas

- **Configuration System**
  - Environment validation
  - Model configuration management

- **Structured Logging**

---

# Project Structure
```
.
├── pyproject.toml
├── README.md
├── src
│   └── ml_project_template
│       ├── api
│       │   ├── app.py
│       │   ├── inference.py
│       │   └── routes
│       │       └── api.py
│       ├── configs
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── model_configs.py
│       │   └── validator.py
│       ├── early_stopping
│       │   └── early_stopping.py
│       ├── errors
│       │   ├── exceptions.py
│       │   └── __init__.py
│       ├── __init__.py
│       ├── logging
│       │   ├── __init__.py
│       │   └── setup.py
│       ├── main.py
│       ├── metrics
│       │   ├── classification
│       │   │   └── metrics.py
│       │   ├── __init__.py
│       │   └── regression
│       │       └── metrics.py
│       ├── model
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── trainer.py
│       └── schemas
│           ├── app_configs.py
│           ├── __init__.py
│           ├── request_schemas.py
│           └── response_schemas.py
└── uv.lock
```

---
# Installation
You can clonse the repository:
```bash
git clone https://github.com/MarwanMohammed2500/ml_project_template
cd ml_project_template
```
and then rename it to what you want

Or you can create a new repo using this one as a template

---
# Dependancies
All dependancies are outlined in pyproject.toml, and can be installed with:
```bash
uv sync
```

---
# Running the API Server
Start the FastAPI Uvicorn server by running:
```bash
uv run -m src.ml_project_template.main
```

---
# Configurations
You can define environment variables and then validate them using `validate_env_vars`, you can also define model-specific configs in `src/ml_project_template/configs/model_configs.py`, which can then be used for model loading, inference, training, etc.

Example Config:
```python
MODEL_PATH = "path/to/model/file"
MODEL_TYPE = "onnx/pt"
TASK_TYPE = "binary/multiclass/regression"
CLASS_MAP = {0: "your", 1: "class", 2: "map"}
DECISION_THRESHOLD = 0.5
```

---
# Training a Model

Use the `SupervisedModelTrainer` class.

Example:

```python
trainer = SupervisedModelTrainer(
    task_type="binary",
    num_epochs=50,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
)

trainer.train()
```

---
# Early Stopping
Example usage:
```python
early_stopper = EarlyStopping(patience=5)

trainer = SupervisedModelTrainer(
    ...
    early_stopper=early_stopper
)
```
Training will stop immediately if validation loss stop improving by a certain value `delta`

---
# Metrics

Classification metrics include:
* Accuracy
* Precision
* Recall
* F1 Score

Regression metrics include:
* MSE
* RMSE
* Normalized RMSE

Metrics are implemented using `torchmetrics`.

---
# Contributions
Feel free to fork the repo and add to it what you feel fits and then open a PR, we'll discuss it and if we reach a conclusion that the additinos are improvements, they'll be added and you'll be listed as a contributor