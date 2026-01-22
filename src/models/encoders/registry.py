import torch
import torch.nn as nn
import os
from omegaconf import (OmegaConf, DictConfig)
from collections import namedtuple
from itertools import product
from ..types import *




_CONFIGS_REGISTRY: Dict[str, Dict[str, Any]] = {"sequential": {}, "visual": {}}
def register_config(name: str, type: str) -> Any:
    def decorator(cls: Any) -> Any:
        if not is_dataclass(cls):
            raise TypeError(f"instance cls is not a dataclass: {type(cls)}, {cls.__mro__}")
        if name in _CONFIGS_REGISTRY:
            raise KeyError("{name} already in configs registry")
        _CONFIGS_REGISTRY[type][name] = cls
        return cls
    return decorator

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

def load_model(config: Union[str, Any]) -> nn.Module:
    if isinstance(config, str):
        config = OmegaConf.load(config)
    return _ENCODERS_REGISTRY[config.name](config)


class FusionManager():

    def __init__(
        self,
        num_heads: Optional[int]=4,
        encoder2include: Optional[List[str]]=["lstm", "vit"],
        checkpoints: Optional[Dict[str, Dict[str, Any]]]={"sequential": {}, "visual": {}, "base": {}},
        apply_last_normalization: Optional[bool]=True,
        randomize_normaliation: Optional[bool]=True,
        attention_activation: Optional[str]="tanh",
        fusion_features_size: Optional[int]=312
    ):
        # print(type(checkpoints))
        self.internel_config: Dict[str, Any] = {
            "num_heads": num_heads,
            "encoders2include": encoder2include,
            "checkpoints": checkpoints,
            "apply_last_normalization": apply_last_normalization,
            "randomize_normalization": randomize_normaliation,
            "fusion_features_size": fusion_features_size,
            "activation_fn": attention_activation
        }
        self.sequential: Dict[str, Any] = {}
        self.visual: Dict[str, Any] = {}
        


    def _back2structured(self, type, dict_cfg) -> Any:
        return _CONFIGS_REGISTRY[type][dict_cfg.name](**dict_cfg)
    
    def build_config(self) -> None:
        self.sequential = _CONFIGS_REGISTRY["sequential"]
        self.visual = _CONFIGS_REGISTRY["visual"]
        self.cfg_splits = {
            "sequential": self.sequential,
            "visual": self.visual
        }

    def save_config(self, path: str, split_mode: Optional[bool]=False) -> None:
        base_cfg = OmegaConf.create(vars(self)["internel_config"])
        # print(type(vars(self)["internel_config"]["checkpoints"]))
        if not split_mode:
            config = OmegaConf.merge({
                "sequential": {k: OmegaConf.structured(c) for (k, c) in self.sequential.items()},
                "visual": {k: OmegaConf.structured(c) for (k, c) in self.visual.items()},
                "base": base_cfg
            })
            OmegaConf.save(config, path)
        
        else:
            assert ("." not in path), \
            (f"try to use split_mode: False for types like: {path}")
            if not os.path.exists(path):
                os.mkdir(path)

            def save_cfgf(configs: Union[Any, Dict[str, Any]], type: str=None): 
                if not isinstance(configs, DictConfig):
                    for (k, c) in configs.items():
                        filename = os.path.join(path, f"{k}-{type}.yaml")
                        OmegaConf.save(c, filename)
                else:
                    filename = os.path.join(path, f"{type}.yaml")
                    OmegaConf.save(configs, filename)

            save_cfgf(self.sequential, "sequential")
            save_cfgf(self.visual, "visual")
            save_cfgf(base_cfg, "base")
            

    def load_config(self, source: Union[str, List[str]]) -> None:
        source_isfile = False
        if isinstance(source, str) and (os.path.isfile(source)):
            config = OmegaConf.load(source)
            self.sequential = {c.name: self._back2structured("sequential", c) for c in config.sequential.values()}
            self.visual = {c.name: self._back2structured("visual", c) for c in config.visual.values()}
            base_cfg = config.base
            source_isfile = True
        
        elif isinstance(source, list) or (not source_isfile):
            cfgs = {"sequential": self.sequential, "visual": self.visual}
            if not source_isfile:
                source = [os.path.join(source, file) for file in os.listdir(source)]
            for path in source:
                c = OmegaConf.load(path)
                if not "base" in path:
                    c = self._back2structured(c.name, c)
                    split = os.path.basename(path).split(".")[0].split("-")[-1]
                    cfgs[split].update({c.name: c})
                else:
                    base_cfg = c
        
        for (key, value) in base_cfg.items():
            if key in self.internel_config:
                self.internel_config[key] = value
            else:
                raise KeyError(f"key error, {key} not specified in self.internel_config")
            
        self.cfg_splits = {
            "sequential": self.sequential,
            "visual": self.visual
        }   
            
            


   











