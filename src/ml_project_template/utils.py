import numpy as np
from typing import Optional, Any
import random
import torch
import mlflow


def set_seed(seed: int = 42):
    """
    sets the seed in all needed libraries (torch, torch.cuda, numpy, and random for general python code)

    Arguments:
    seed:int=42, the seed
    """
    np.random.seed(seed)  # Sets NumPy random seed
    torch.manual_seed(seed)  # type:ignore - Sets PyTorch's random seed
    torch.cuda.manual_seed(seed)  # Sets PyTorch's random seed on CUDA
    torch.cuda.manual_seed_all(
        seed
    )  # Sets PyTorch's random seed on all objects on the GPU
    random.seed(seed)  # Sets the random seed on all python objects


def export_onnx_from_torch(
    input_sample: Any,
    output_path: str,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_shapes: Optional[dict[str, dict[int, str]]] = None,
    model_path: Optional[str] = None,
    model_uri: Optional[str] = None,
):
    """
    Exports a PyTorch model to ONNX format.

    Arguments:
    model: torch.nn.Module - The PyTorch model to export.
    input_sample: torch.Tensor - A sample input tensor to trace the model's computation graph.
    output_path: str - The file path where the ONNX model will be saved.
    """
    model_to_export: torch.nn.Module
    if model_path is not None:
        model_to_export = torch.load(model_path)
    elif model_uri is not None:
        model_to_export = mlflow.pytorch.load_model(model_uri)  # type: ignore
    else:
        raise ValueError("Either `model_path`, or `model_uri` should be provided.")

    model_to_export = model_to_export.cpu()  # type: ignore
    input_sample = input_sample.cpu()
    model_to_export.eval()  # type: ignore

    # Export the model to ONNX format
    torch.onnx.export(  # type: ignore
        model_to_export,
        input_sample,
        output_path,
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
    )
