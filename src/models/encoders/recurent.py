import torch 
import torch.nn as nn
import torch.nn.functional as F

from .registry import (
    register_encoder, 
    get_activation, 
    register_config
)
from ..types import *


@register_config("lstm", "sequential")
@dataclass
class RecurrentEncoderConfig:
    input_features: Optional[int]=128
    name: Optional[str]="lstm"
    data_domain: Optional[str]="temporal"
    hiden_features_size: Optional[int]=128
    out_features_size: Optional[int]=None
    activation: Optional[str]="sigmoid"
    normalization: Optional[bool]=True
    random_normalization: Optional[bool]=True
    num_layers: Optional[int]=3
    add_bias: Optional[bool]=True
    bidirectional: Optional[bool]=False
    images_size: Tuple[int, int]=(112, 448)
    patch_size: Tuple[int, int]=(16, 32)


        

class AdaptiveLayerNormalization(nn.Module):
    def __init__(
        self, 
        hiden_features_size: Optional[int]=None,
        normal_randomization: Optional[bool]=False,
        cfg: Optional[RecurrentEncoderConfig]=None
    ) -> None:
        super(AdaptiveLayerNormalization, self).__init__()
        self.cfg = cfg
        if self.cfg is not None:
            self.C = (
                hiden_features_size 
                if hiden_features_size is not None
                else self.cfg.hiden_features_size
            )
            randomize = (
                normal_randomization
                if normal_randomization is not None
                else self.cfg.random_normalization
            )
        else:
            self.C = hiden_features_size
            randomize = normal_randomization

        assert (self.C is not None), \
        ("ither hiden_features or general model config must be passed")

        self.shift = nn.Parameter(torch.zeros(self.C))
        self.scale = nn.Parameter(torch.ones(self.C))
        if randomize:
            nn.init.normal_(self.shift, 0.0, 1.0)
            nn.init.normal_(self.scale, 0.0, 1.0)
    
    def forward(self, input: SequenceTensor) -> SequenceTensor:
        features = self.scale.view(1, 1, self.C) * input + self.shift.view(1, 1, self.C)
        features = F.sigmoid(features)
        return features

@register_encoder("lstm")
class RecurrentEncoder(nn.Module):
    def __init__(self, cfg: RecurrentEncoderConfig) -> None:
        super(RecurrentEncoder, self).__init__()
        self.cfg = cfg
        self.patches_n = (
            self.cfg.images_size[0] // self.cfg.patch_size[0],
            self.cfg.images_size[1] // self.cfg.patch_size[1]
        )
        self.lstm = nn.LSTM(self.cfg.input_features,
                            self.cfg.hiden_features_size,
                            self.cfg.num_layers,
                            bias=self.cfg.add_bias)
        self.act = get_activation(self.cfg.activation)
        Oc = (
            self.cfg.out_features_size
            if self.cfg.out_features_size is not None
            else self.cfg.hiden_features_size
        )
        if self.cfg.out_features_size is not None:
            self.last_featurees = nn.Linear(self.cfg.hiden_features_size, Oc)
        if self.cfg.normalization:
            self.adanorm = AdaptiveLayerNormalization(Oc, cfg=cfg)
        
        print(self.patches_n)
        self.pooling = nn.AdaptiveAvgPool1d(self.patches_n[0] * self.patches_n[1])
        
    def forward(self, input: SequenceTensor) -> torch.Tensor:
        
        features, (_, _) = self.lstm(input)
        features = self.act(features)
        if self.cfg.normalization:
            features = self.adanorm(features)
        
        features = features.transpose(-1, -2)
        features = self.pooling(features).transpose(-1, -2)
        return features
    



# if __name__ == "__main__":

    # cfg = RecurrentEncoderConfig(input_features=128)
    # encoder = RecurrentEncoder(cfg)
    # test = torch.normal(0, 1, (10, 32, 128))
    # output = encoder(test)
    # print(output.size())