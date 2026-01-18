import torch 
import numpy as np
from typing import (
    Optional, Union,
    List, Dict,
    Any, Tuple,
    NamedTuple
)
from torchtyping import TensorType
from dataclasses import (dataclass, field)



__all__ = [
    "NamedTuple", "PatchTensor", 
    "Optional", "Union",
    "List", "Dict",
    "Any", "Tuple",
    "ImageTensor", "SequenceTensor",
    "dataclass", "field"
]
ImageTensor = TensorType["B", "C", "W", "H"]
SequenceTensor = TensorType["B", "N", "C"]
PatchTensor = Union[SequenceTensor, ImageTensor]




