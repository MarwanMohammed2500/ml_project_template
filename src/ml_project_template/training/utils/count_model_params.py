import torch
from torch import nn
import pandas as pd
from typing import Any


def count_model_parameters(
    model: nn.Module, example_input: torch.Tensor
) -> pd.DataFrame:
    layer_summaries: list[dict[str, Any]] = []

    def log_input_shape(
        module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        name = module.__class__.__name__
        num_params = sum(p.numel() for p in module.parameters())
        layer_info: dict[str, Any] = {
            "layer": name,
            "input_shape": list(input[0].shape),
            "output_shape": list(output[0].shape),
            "num_params": num_params,
        }
        layer_summaries.append(layer_info)

    # Register hook on each layer
    for name, module in model.named_modules():  # type: ignore
        module.register_forward_hook(log_input_shape)  # type: ignore

    with torch.inference_mode():
        print(f"------- Input Shape -------\n{example_input.shape}")
        model(example_input)

    df = pd.DataFrame(layer_summaries)
    df.to_json("model_summary.json", orient="records", indent=4)
    return df
