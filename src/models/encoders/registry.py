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
        return cls
    return decorator

@dataclass 
class GeneralEncoderConfig:
    loaded_configs: Dict[str, Dict[str, Any]]=field(default_factory=lambda: {
        "sequential": {}, 
        "visual": {},
        "base": {}
    })
    
    def __post_init__(self) -> None:
        self.build_config()

    def get_configs(self, configs2get: List[str]) -> Dict[str, Any]:
        configs = {}
        configs.update({"sequential": [
            cfg 
            for (name, cfg) in self.loaded_configs["sequential"].items()
            if name in configs2get
        ]})
        configs.update({"visual": [
            cfg 
            for (name, cfg) in self.loaded_configs["visual"].items()
            if name in configs2get
        ]})
        configs.update({"base": self.loaded_configs["base"]})
        return configs

    def _back2structured(self, name, dict_cfg) -> Any:
        return _CONFIGS_REGISTRY[name](**dict_cfg)
    
    def build_config(self) -> None:
        for (name, cfg) in _CONFIGS_REGISTRY.items():
            if name == "base":
                continue
            type = name.split("-")[1]
            self.loaded_configs[type].update({name: cfg})
        
        cfg = _CONFIGS_REGISTRY["base"]
        self.loaded_configs.update({"base": cfg})
    
    def save_config(self, path: str, split_mode: Optional[bool]=False) -> None:
        if not split_mode:
            config_part = lambda cfg_type: {
                name.replace("-", "_"): OmegaConf.structured(cfg)
                for (name, cfg) in self.loaded_configs[cfg_type].items()
            }
            config = OmegaConf.merge({
                "sequential": config_part("sequential"), 
                "visual": config_part("visual"),
                "base": OmegaConf.structured(self.loaded_configs["base"])
            })
            OmegaConf.save(config, path)
        
        else:
            assert ("." not in path), \
            (f"try to use split_mode: False for types like: {path}")
            if not os.path.exists(path):
                os.mkdir(path)

            for (name, cfg) in list(self.loaded_configs["sequential"].items()) \
                    + list(self.loaded_configs["visual"].items()) \
                    + [("base", self.loaded_configs["base"]), ]:
                file = os.path.join(path, f"{name.replace("-", "_")}.yaml")
                print(file)
                cfg = OmegaConf.structured(cfg)
                OmegaConf.save(cfg, file)
            

    def load_config(self, source: Union[str, List[str]]) -> None:
        source_isfile = False
        if isinstance(source, str) and (os.path.isfile(source)):
            config = OmegaConf.load(source)
            self.loaded_configs["sequential"] = {
                name.replace("_", "-"): self._back2structured(name.replace("_", "-"), cfg) 
                for (name, cfg) in config.sequential.items()
            }
            self.loaded_configs["visual"] = {
                name.replace("_", "-"): self._back2structured(name.replace("_", "-"), cfg) 
                for (name, cfg) in config.visual.items()
            }
            self.loaded_configs["base"] = self._back2structured("base", config.base)
            source_isfile = True
        
        elif isinstance(source, list) or (not source_isfile):
            if not source_isfile:
                source = [os.path.join(source, file) for file in os.listdir(source)]
            for path in source:
                fname = os.path.basename(path).split(".")[0]
                if "base" not in fname:
                    fname = fname.replace("_", "-")
                cfg = self._back2structured(fname, OmegaConf.load(path))
                self.loaded_configs.update({fname: cfg})
                











   
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










