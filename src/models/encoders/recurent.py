import torch 
import torch.nn as nn
import torch.nn.functional as F

from .base import (register_encoder, get_activation)
from ..types import *




class RecurrentEncoderOutput(NamedTuple):
    last_activation: SequenceTensor
    last_activation_normalized: SequenceTensor
    last_cell_activation: SequenceTensor
    last_hid_activation: SequenceTensor

@dataclass
class RecurrentEncoderConfig:
    input_features: int
    hiden_features_size: Optional[int]=128
    activation: Optional[str]="sigmoid"
    normalization: Optional[bool]=True
    random_normalization: Optional[bool]=True
    num_layers: Optional[int]=10
    add_bias: Optional[bool]=True
    bidirectional: Optional[bool]=False


_RECURRENT_ENCODER_CONFIGURATIONS = {
    "tiny": RecurrentEncoderConfig(
        input_features=10000,
        hiden_features_size=312,
        normalization=True,
        random_normalization=True,
        num_layers=3,
        bidirectional=False
    ),
    "big":  RecurrentEncoderConfig(
        input_features=128,
        hiden_features_size=512,
        normalization=True,
        random_normalization=True,
        num_layers=6,
        bidirectional=False
    ),
    "large":  RecurrentEncoderConfig(
        input_features=128,
        hiden_features_size=718,
        normalization=True,
        random_normalization=True,
        num_layers=10,
        bidirectional=False
    ),
}


class AdaptiveLayerNormalization(nn.Module):
    def __init__(self, cfg: RecurrentEncoderConfig) -> None:
        super(AdaptiveLayerNormalization, self).__init__()
        self.cfg = cfg

        self.shift = nn.Parameter(torch.zeros(self.cfg.hiden_features_size))
        self.scale = nn.Parameter(torch.ones(self.cfg.hiden_features_size))
        if self.cfg.random_normalization:
            nn.init.normal_(self.shift, 0.0, 1.0)
            nn.init.normal_(self.scale, 0.0, 1.0)
    
    def forward(self, input: SequenceTensor) -> SequenceTensor:
        C = self.cfg.hiden_features_size
        features = self.scale.view(1, 1, C) * input + self.shift.view(1, 1, C)
        features = F.sigmoid(features)
        return features

@register_encoder("lstm-encoder")
class RecurrentEncoder(nn.Module):
    def __init__(self, cfg: RecurrentEncoderConfig) -> None:
        super(RecurrentEncoder, self).__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(self.cfg.input_features,
                            self.cfg.hiden_features_size,
                            self.cfg.num_layers,
                            bias=self.cfg.add_bias)
        self.act = get_activation(self.cfg.activation)
        if self.cfg.normalization:
            self.adanorm = AdaptiveLayerNormalization(cfg)
        
    def forward(self, input: SequenceTensor) -> RecurrentEncoderOutput:
        
        features, (hidden_states, cell_states) = self.lstm(input)
        features = self.act(features)
        features_norm = None
        if self.cfg.normalization:
            features_norm = self.adanorm(features)
        
        if self.cfg.bidirectional:
            last_hiden_state = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)
            last_cell_state = torch.cat([cell_states[-1], cell_states[-2]], dim=-1)
        
        else:
            last_hiden_state = hidden_states[-1]
            last_cell_state = cell_states[-1]

        return RecurrentEncoderOutput(features,
                                      features_norm,
                                      last_cell_state,
                                      last_hiden_state)
    


if __name__ == "__main__":

    cfg = _RECURRENT_ENCODER_CONFIGURATIONS["tiny"]
    model = RecurrentEncoder(cfg)
    print(sum([p.numel() for p in model.parameters()]))