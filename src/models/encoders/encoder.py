import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

from omegaconf import OmegaConf
from .registry import ( 
    load_model, 
    load_configs, 
    get_activation
)
from .registry import GeneralEncoderConfig, register_config, _CONFIGS_REGISTRY
from .recurent import (RecurrentEncoderConfig, AdaptiveLayerNormalization)
from .visual import (VITTransformerConfig, VITSelfAttention)
from ..types import *
from itertools import product


class ECGEncoderOutput(NamedTuple):
    last_fusion: SequenceTensor
    intermediate_fusions: List[SequenceTensor]




class ECGEncoder(nn.Module):
    def __init__(self, base_cfg, encoders_cfg: Dict[str, Any]) -> None:
        super(ECGEncoder, self).__init__()
        self.cfg = base_cfg

        (seq_encoders_cfg, vis_encoders_cfg) = load_configs(encoders_cfg)
        hiden_sizes = set([cfg.hiden_features_size for cfg in seq_encoders_cfg.values()] + 
                          [cfg.hidden_features_size for cfg in vis_encoders_cfg.values()])
        assert len(hiden_sizes) == 1, \
        (f"not all models have the same hiden features dimentions: {hiden_sizes}")
        hiden_d = hiden_sizes.pop()

        self.sequential_encoders = nn.ModuleDict({
            key: load_model(cfg, key) 
            for (key, cfg) in seq_encoders_cfg.items()
        })
        num_seq_m = len(self.sequential_encoders)
        self.visual_encoders = nn.ModuleDict({
            key: load_model(cfg, key) 
            for (key, cfg) in vis_encoders_cfg.items()
        })
        num_vis_m = len(self.visual_encoders)
        self.features = nn.Sequential(
            nn.Linear(num_seq_m * num_vis_m * hiden_d, self.cfg.output_features_size),
            get_activation("sigmoid")
        )
        self.adanorm = AdaptiveLayerNormalization(self.cfg.output_features_size)
        self.attention_layers = nn.ModuleList([
            VITSelfAttention(
                activation_fn=self.cfg.attention_activation,
                hiden_features_size=hiden_d
            ) for _ in range(num_seq_m * num_seq_m)
        ])
    
    def forward(self, signal: SequenceTensor, spec: ImageTensor) -> ECGEncoderOutput:
        
        sequential_acts = {}
        xs = signal
        for key, model in self.sequential_encoders.items():
            xs = model(sequential_acts).last_activation
            sequential_acts[key] = xs
        
        visual_acts = {}
        xv = spec
        for key, model in self.visual_encoders.items():
            xv = model(xv).features
            visual_acts[key] = xv
        
        pares = product(list(sequential_acts.keys()), list(visual_acts.keys()))
        fusions = []
        for idx, (xs, xv) in enumerate(pares):
            fusion = self.attention_layers[idx](xs, xv)
            fusions.append(fusion)
        
        fusion = torch.cat(fusion, dim=-1)
        fusion_features = self.features(fusion)
        fusion_features_normalized = self.adanorm(fusion_features)
        return ECGEncoderOutput(fusion_features_normalized,
                                fusions)
            
        


if __name__ == "__main__":

    config = GeneralEncoderConfig()
    print(_CONFIGS_REGISTRY)
    config.build_config()
    config.save_config("test.yaml")        

# if __name__ == "__main__":

#    pass
    # base_cfg = ECGEncoderConfig()
    # rnn_cfg = RecurrentEncoderConfig(100)
    # vit_cfg = VITTransformerConfig()
    # configs = [rnn_cfg, vit_cfg]
    # model = ECGEncoder(base_cfg, configs)
    
    # print(sum([p.numel() for p in model.parameters()]))








