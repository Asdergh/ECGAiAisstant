import torch
import torch.nn as nn
import os
from omegaconf import OmegaConf
from ..types import *




_CONFIGS_REGISTRY: Dict[str, Any] = {}
def register_config(name: str) -> Any:
    def decorator(cls: Any) -> Any:
        if not is_dataclass(cls):
            raise TypeError(f"instance cls is not a dataclass: {type(cls)}, {cls.__mro__}")
        if name in _CONFIGS_REGISTRY:
            raise KeyError("{name} already in configs registry")
        _CONFIGS_REGISTRY[name] = cls
        print("reg worked")
        return cls
    return decorator

@dataclass 
class GeneralEncoderConfig:
    _configs2include: List[str]=field(default_factory=lambda: ["lstm-sequential-encoder", 
                                                       "vit-visual-encoder", 
                                                       "base"])

    def build_config(self, configs2include: Optional[List[str]]=None) -> None:
        if configs2include is not None:
            self._configs2include = list(set(self._configs2include + configs2include))
        for name in self._configs2include:
            cfg = _CONFIGS_REGISTRY[name]
            setattr(self, name.replace("-", "_"), cfg)
            print(hasattr(self, name))
    
    def save_config(self, path: str) -> None:
        config = OmegaConf.merge({
            name: OmegaConf.structured(cfg)
            for (name, cfg) in _CONFIGS_REGISTRY.items()
        })
        OmegaConf.save(config, path)
    
    def load_config(self, path: str) -> None:
        config = OmegaConf.load(path)
        for (name, cfg) in config.items():
            if name not in self._configs2include:
                continue
            cfg = _CONFIGS_REGISTRY[name](**cfg)
            cfg = OmegaConf.structured(cfg)
            setattr(self, name, cfg) 


@register_config("base")
@dataclass
class ECGEncoderConfig:
    num_heads: Optional[int]=4
    apply_last_normalization: Optional[bool]=True
    randomize_normaliation: Optional[bool]=True
    output_features_size: Optional[int]=312
    attention_activation: Optional[str]="tanh"


@register_config("lstm-sequential-encoder")
@dataclass
class RecurrentEncoderConfig:
    input_features: int
    name: Optional[str]="lstm-sequential-encoder"
    data_domain: Optional[str]="temporal"
    hiden_features_size: Optional[int]=128
    activation: Optional[str]="sigmoid"
    normalization: Optional[bool]=True
    random_normalization: Optional[bool]=True
    num_layers: Optional[int]=10
    add_bias: Optional[bool]=True
    bidirectional: Optional[bool]=False

@register_config("vit-visual-encoder")
@dataclass
class VITTransformerConfig:
    name: Optional[str]="vit-visual-encoder"
    data_domain: Optional[str]="visual"
    input_type: Optional[str]="signal"
    image_size: Tuple[int, int]=(224, 112)
    patch_size: Tuple[int, int]=(16, 16)
    num_transformer_blocks: Optional[int]=4
    out_hidden_indices: Optional[List[int]]=field(default_factory=lambda: [0, 1, 2])
    embeddings_size: Optional[int]=128
    hidden_features_size: Optional[int]=128
    backbone_activation: Optional[str]="tanh"
    transformer_activation: Optional[str]="relu"
    attention_activation: Optional[str]="sigmoid"
    attention_pool_scale: Optional[int]=2


   
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


def load_configs(configs: Union[str, List[str]]):
    if isinstance(configs, str):
        configs = [
            os.path.join(configs, fname) 
            for fname in os.path.listdir(configs) 
            if "encoder" in fname
        ]
    
    build_cfg = lambda cfg: (
        OmegaConf.load(cfg) 
        if isinstance(cfg, str)
        else OmegaConf.structured(cfg)
    )
    sequential_encoders_cfg = {}
    visual_encoders_cfg = {}
    for config in configs:
        cfg = build_cfg(config)
        if "sequential" in cfg.name:
            sequential_encoders_cfg.update({cfg.name: cfg})
        elif "visual" in cfg.name:
            visual_encoders_cfg.update({cfg.name: cfg})
        else:
            raise TypeError(f"unknown config type for encoders: {cfg.name}")
    
    return (sequential_encoders_cfg, visual_encoders_cfg)

def load_model(config: Union[str, Any], key: Optional[str]=None) -> nn.Module:

    if isinstance(config, str):
        config = OmegaConf.load(config)
    name = (key if key is not None else config.name)
    return _ENCODERS_REGISTRY[name](config)


if __name__ == "__main__":

    config = GeneralEncoderConfig()
    print(_CONFIGS_REGISTRY)
    config.build_config()
    config.save_config("test.yaml")
    print(config.base, config.lstm_sequential_encoder)





