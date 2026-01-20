import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

from omegaconf import OmegaConf
from .registry import GeneralEncoderConfig
from .recurent import (RecurrentEncoderConfig, AdaptiveLayerNormalization)
from .visual import (VITTransformerConfig, VITSelfAttention)
from .registry import ( 
    load_model,  
    get_activation,
    register_config,
    GeneralEncoderConfig
)
from ..types import *
from itertools import product


class ECGEncoderOutput(NamedTuple):
    last_fusion: SequenceTensor
    intermediate_fusions: List[SequenceTensor]





@register_config("base")
@dataclass
class ECGEncoderConfig:
    name: Optional[str]="base"
    num_heads: Optional[int]=4
    encoder2include: Optional[List[str]]=field(default_factory=lambda: [
        "lstm-sequential-encoder", 
        "vit-visual-encoder"
    ])
    apply_last_normalization: Optional[bool]=True
    randomize_normaliation: Optional[bool]=True
    output_features_size: Optional[int]=312
    attention_activation: Optional[str]="tanh"

class ECGEncoder(nn.Module):
    def __init__(self, general_cfg: GeneralEncoderConfig) -> None:
        super(ECGEncoder, self).__init__()

        self.base_cfg = general_cfg.load_config["base"]
        configs = general_cfg.get_configs(self.base_cfg.encoder2include)
        self.sequential_models = {cfg.name: load_model(cfg) for cfg in configs["sequential"]}
        self.visual_models = {cfg.name: load_model(cfg) for cfg in configs["visual"]}

        hiden_d = self.check_hiden_dims(configs["sequential"], configs["visual"])
        (num_seq_m, num_vis_m) = (len(self.sequential_models), len(self.visual_models)) 
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
    
    def check_hiden_dims(self, sq_cfgs: Dict[str, Any], vis_cfgs: Dict[str, Any]):

        hiden_sizes = set([cfg.hiden_features_size for cfg in sq_cfgs] + 
                          [cfg.hiden_featuers_size for cfg in vis_cfgs])
        assert (len(hiden_sizes) == 1), \
        ("the hiden sizes nfrof different models are not the same")
        return hiden_sizes.pop() 
    
    # def _load_models(self, config) -> Tuple[nn.ModuleDict, nn.ModuleDict]:
        
    
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
    model = ECGEncoder(config)
    # config.save_config("test.yaml")        

# if __name__ == "__main__":

#    pass
    # base_cfg = ECGEncoderConfig()
    # rnn_cfg = RecurrentEncoderConfig(100)
    # vit_cfg = VITTransformerConfig()
    # configs = [rnn_cfg, vit_cfg]
    # model = ECGEncoder(base_cfg, configs)
    
    # print(sum([p.numel() for p in model.parameters()]))








