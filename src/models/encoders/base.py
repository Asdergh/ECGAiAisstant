import torch
import torch.nn as nn

from abc import (ABC, abstractmethod)
from ..types import *
from functools import wraps


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