import torch 
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

from omegaconf import OmegaConf
from .recurent import (RecurrentEncoderConfig, AdaptiveLayerNormalization)
from .visual import (VITTransformerConfig, VITSelfAttention)
from .registry import (FusionManager, load_model, get_activation, _ENCODERS_REGISTRY)
from ..types import *
from itertools import product







class FCBlock(nn.Module):

    def __init__(
        self,
        in_features_size: Optional[int]=None,
        out_features_size: Optional[int]=None,
        dropout_rate: Optional[float]=0.45,
        activation_fn: Optional[str]="relu"
    ) -> None:
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(in_features_size, out_features_size)
        self.act_fn = get_activation(activation_fn)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features_size, out_features_size)
    
    def forward(self, input: SequenceTensor) -> SequenceTensor:
        x = self.fc1(input)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class FusionEncoder(nn.Module):
    def __init__(self, source_cfg: Optional[str]=None) -> None:
        super(FusionEncoder, self).__init__()
        self.manager = FusionManager()
        if (source_cfg is not None) \
            and (os.path.exists(source_cfg)):
            self.manager.load_config(source_cfg)
        else:
            # print("CONFIGS WAS BUILEDE")
            self.manager.build_config()
        
        self.cfg = self.manager.internel_config
    
    
    def build_encoders(self, split_type) -> nn.ModuleDict:
        models = {}
        ckpt_pkg = self.cfg["checkpoints"][split_type]
        for (k, c) in self.manager.cfg_splits[split_type].items():
            model = load_model(c)
            if (ckpt_pkg is not None) \
                and (k in ckpt_pkg):
                if os.path.exists(ckpt_pkg[k]):
                    model.load_state_dict(torch.load(ckpt_pkg[k], weights_only=True))
                    model.eval()
                else:
                    raise ValueError(f"counld find ckepoints at location: {ckpt_pkg[k]}")
            models.update({k: model})
        return nn.ModuleDict(models)
        
    def build_fusion_encoder(self):
        #build registered models with there checkpoints
        self.sequential_models = self.build_encoders("sequential")
        self.visual_models = self.build_encoders("visual")
        self._models = {
            "sequential": self.sequential_models,
            "visual": self.visual_models
        }
        ckpt_pkg = self.cfg["checkpoints"]["base"]

        #build projections to project all features into fusion domain
        def get_projections(split_type: str):
            projection_blocks = {}
            cfgs = self.manager.cfg_splits[split_type]
            for k in cfgs.keys():
                hiden_d = (
                    cfgs[k].out_features_size
                    if cfgs[k].out_features_size is not None
                    else cfgs[k].hiden_features_size
                )
                projection = FCBlock(hiden_d, self.cfg["fusion_features_size"])
                if k in ckpt_pkg:
                    projection.load_state_dict(torch.load(ckpt_pkg[k], weights_only=True))
                projection_blocks.update({k: projection})
            return projection_blocks
        sequential_pj = get_projections("sequential")
        visual_pj = get_projections("visual")
        self.projections = nn.ModuleDict(sequential_pj | visual_pj)

        #build fusion blocks for calculate the correlation between activations
        fused_N = len(self.sequential_models) + len(self.visual_models)
        self.fusion_gates = nn.Parameter(torch.normal(0, 1, (fused_N * self.cfg["fusion_features_size"], )))
        self.fusion = VITSelfAttention("tanh", fused_N *  self.cfg["fusion_features_size"])
        self.activation = get_activation(self.cfg["activation_fn"])

    
    def forward(self, seq_input: SequenceTensor, freq_input: ImageTensor) -> Dict[str, Any]:
        
        def forward_throught(input: torch.Tensor, split_type: str):
            activations = []
            for (k, model) in self._models[split_type].items():
                x = model(input)
                print(k, x.size())
                x = self.projections[k](x)
                activations.append(x)
            return (
                activations[0]
                if len(activations) == 1
                else torch.stack(activations, dim=-1)
            )
        
        sequentials = forward_throught(seq_input, "sequential")
        visuals = forward_throught(freq_input, "visual")
        features_vs = torch.cat([visuals, sequentials], dim=-1)
        features_sv = torch.cat([sequentials, visuals], dim=-1)
        
        fusion_vs = self.fusion(features_vs)
        fusion_sv = self.fusion(features_sv)
        fused_features = self.fusion_gates * fusion_vs + (1 - self.fusion_gates) * fusion_sv
        return fused_features
        
        
        

        




if __name__ == "__main__":
    
    # model = FusionEncoder()
    # model.manager.build_config()
    # model.manager.save_config("test.yaml")
    model = FusionEncoder("test.yaml")
    model.build_fusion_encoder()
    test_sq = torch.normal(0, 1, (10, 1000, 128))
    test_specs = torch.normal(0, 1, (10, 1, 224, 448))
    output = model(test_sq, test_specs)
    print(output.size())






    # model.manager.build_config()
    # model.manager.save_config("test.yaml")
    # model.build_fusion_encoder()
    # print(model)
    # model.manager.save_config("test.yaml")
    # print(model)
    # print(sum([p.numel() for p in model.parameters()]))
    # print(_ENCODERS_REGISTRY)
    # encoder_builder = EncoderBuilder()
    # encoder_builder.save_config("test.yaml", split_mode=False)
    # encoder_builder.save_config("test_split", split_mode=True)
    # encoder_builder.load_config("test.yaml")
    # print(encoder_builder.visual)
    # print(encoder_builder.sequential)
    # print(encoder_builder.internel_config)
    
    
   
    # config.save_config("test.yaml")        

# if __name__ == "__main__":

#    pass
    # base_cfg = ECGEncoderConfig()
    # rnn_cfg = RecurrentEncoderConfig(100)
    # vit_cfg = VITTransformerConfig()
    # configs = [rnn_cfg, vit_cfg]
    # model = ECGEncoder(base_cfg, configs)
    
    # print(sum([p.numel() for p in model.parameters()]))








