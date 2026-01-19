import torch 
import torch.nn as nn
import torch.nn.functional as F

from .registry import (
    register_encoder, 
    get_activation, 
    RecurrentEncoderConfig
)
from ..types import *



print("RECURENT FILE IF DEBUGED")
class RecurrentEncoderOutput(NamedTuple):
    last_activation: SequenceTensor
    last_activation_normalized: SequenceTensor
    last_cell_activation: SequenceTensor
    last_hid_activation: SequenceTensor




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
            self.C = self.cfg.hiden_features_size
            randomize = self.cfg.random_normalization
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

@register_encoder("lstm-sequential-encoder")
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
            self.adanorm = AdaptiveLayerNormalization(cfg=cfg)
        
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