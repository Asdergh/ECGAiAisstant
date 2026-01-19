import torch 
import numpy as np
from typing import (
    Optional, Union,
    List, Dict,
    Any, Tuple,
    NamedTuple
)
from torchtyping import TensorType
from dataclasses import (dataclass, field, is_dataclass)



__all__ = [
    "NamedTuple", "PatchTensor", 
    "Optional", "Union",
    "List", "Dict",
    "Any", "Tuple",
    "ImageTensor", "SequenceTensor",
    "dataclass", "field",
    "is_dataclass"
]
ImageTensor = TensorType["B", "C", "W", "H"]
SequenceTensor = TensorType["B", "N", "C"]
PatchTensor = Union[SequenceTensor, ImageTensor]




