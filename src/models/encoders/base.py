import torch
import torch.nn as nn

from abc import (ABC, abstractmethod)
from ..types import *
from functools import wraps



_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}
get_activation = lambda activation: (
    nn.Identity() if activation not in _ACTIVATIONS
    else _ACTIVATIONS[activation](dim=-1) if activation == "softmax"
    else _ACTIVATIONS[activation]()
)

_ENCODERS_REGISTRY = {}
def register_encoder(name: str):
    def decorator(cls: Any) -> Any:
        if not (nn.Module in cls.__mro__):
            raise TypeError(f"unknow model type: {type(cls)}")
        if name in _ENCODERS_REGISTRY:
            raise KeyError(f"encoder: {name} if already registerd")
        _ENCODERS_REGISTRY[name] = cls
        return cls
    return decorator