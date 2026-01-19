import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from .base import (register_encoder, get_activation)
from ..types import *





_IN_CHANNELS = {
    "image": 3,
    "signal": 1
}

class VITBlockOutput(NamedTuple):
    features: PatchTensor
    film_norm: PatchTensor
    attention: PatchTensor
    

@dataclass
class VITTransformerOutput:
    patches_n: Tuple[int, int]
    last_activation: SequenceTensor
    intermediate_activations: List[SequenceTensor]
    attention_activations: List[SequenceTensor]
    
    
    def _sequence2spatial(self, tokens: PatchTensor) -> PatchTensor:
        tokens = tokens.transpose(-1, -2)
        spatial_tokens = tokens.view(*tokens.shape[:-1], *self.patches_n)
        return spatial_tokens


    @property
    def get_spatial_tokens(self):
        return VITTransformerOutput(
            patches_n=self.patches_n,
            last_activation=self._sequence2spatial(self.last_activation),
            intermediate_activations=[
                self._sequence2spatial(val) 
                for val in self.intermediate_activations
            ],
            attention_activations=[
                self._sequence2spatial(val)
                for val in self.attention_activations
            ]
        )
        

@dataclass
class VITTransformerConfig:
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



_VIT_CONFIGURATIONS = {
    "tiny":  VITTransformerConfig(
                image_size=(224, 224),
                patch_size=(16, 16),
                hidden_features_size=312,
                num_transformer_blocks=4,
                attention_pool_scale=2
            ),
    "big":  VITTransformerConfig(
                image_size=(448, 448),
                patch_size=(32, 32),
                hidden_features_size=562,
                num_transformer_blocks=4,
                attention_pool_scale=2
            ),
    "large":  VITTransformerConfig(
                image_size=(896, 896),
                patch_size=(64, 64),
                hidden_features_size=758,
                num_transformer_blocks=4,
                attention_pool_scale=2
            )
}

class PatchEmbedder(nn.Module):
    def __init__(self, cfg: VITTransformerConfig) -> None:
        super(PatchEmbedder, self).__init__()
        self.cfg = cfg
        assert ((self.cfg.image_size[0] % self.cfg.patch_size[0]) == 0
                and (self.cfg.image_size[1] % self.cfg.patch_size[1]) == 0), \
                (f"images dims: [{self.cfg.image_size}] must be dividable to patch dims: [{self.cfg.patch_size}]")
        
        C = self.cfg.hidden_features_size
        self.embeddings = nn.Conv2d(_IN_CHANNELS[self.cfg.input_type], 
                                   C, 1, self.cfg.patch_size, 0)
        self.features = nn.Sequential(
            nn.Linear(C, C),
            get_activation(self.cfg.backbone_activation)
        )
    
    def forward(self, input: ImageTensor) -> SequenceTensor:
        x = input
        spatial_embeddings = self.embeddings(x)
        sequence_embedding = torch.flatten(spatial_embeddings, -2).permute(0, 2, 1)
        features = self.features(sequence_embedding)
        return features



class VITSelfAttention(nn.Module):
    def __init__(self, cfg: VITTransformerConfig) -> None:
        super(VITSelfAttention, self).__init__()
        self.cfg = cfg

        self.d_scale = (math.sqrt(self.cfg.hidden_features_size)) ** -1
        self.C = self.cfg.hidden_features_size
        (self.Pw, self.Ph) = (self.cfg.image_size[0] // self.cfg.patch_size[0], 
                              self.cfg.image_size[1] // self.cfg.patch_size[1])
        self.quary = nn.Sequential(nn.Linear(self.C, self.C), get_activation(self.cfg.attention_activation))
        self.key = nn.Sequential(nn.Linear(self.C, self.C), get_activation(self.cfg.attention_activation))
        self.value = nn.Sequential(nn.Linear(self.C, self.C), get_activation(self.cfg.attention_activation))
    
    def _sequence2spatial(self, tokens: SequenceTensor) -> PatchTensor:
        B = tokens.size(0)
        tokens = tokens.transpose(-1, -2)
        spatial_tokens = tokens.view(B, self.C, self.Pw, self.Ph)
        return spatial_tokens

    def _spatial2sequence(self, tokens: PatchTensor) -> SequenceTensor:
        sequence_tokens = torch.flatten(tokens, start_dim=-2)
        sequence_tokens = sequence_tokens.permute(0, 2, 1)
        return sequence_tokens
    
    def forward(self, input: SequenceTensor) -> SequenceTensor:
        
        Q, K, V = (
            self._sequence2spatial(self.quary(input)), 
            self._sequence2spatial(self.key(input)), 
            self._sequence2spatial(self.value(input))
        )
        if self.cfg.attention_pool_scale is not None:
            q = F.avg_pool2d(Q, (self.cfg.attention_pool_scale, 1))
            k = F.avg_pool2d(K, (self.cfg.attention_pool_scale, 1))
            
            Qk = self.d_scale * torch.einsum("ncik, nckj -> ncij", Q, k.transpose(-1, -2))
            Qk = F.softmax(Qk, dim=-1)
           
            Kq = self.d_scale * torch.einsum("ncik, nckj -> ncij", q, K.transpose(-1, -2))
            Kq = F.softmax(Kq, dim=-1)
            
            Kqv = torch.einsum("ncik, nckj -> ncij", Kq, V)
            QkKqv = torch.einsum("ncik, nckj -> ncij", Qk, Kqv)
            return self._spatial2sequence(QkKqv)

        else:
            QK = self.d_scale * torch.einsum("ncik, nckj -> ncij", Q, K)
            weights = F.softmax(QK, dim=-1)
            QKV = torch.einsum("ncik, nckj -> ncij", weights, V)
            return self._spatial2sequence(QKV)



@register_encoder("vit-encoder")
class VITTransformerBlock(nn.Module):
    def __init__(self, cfg: VITTransformerConfig) -> None:
        super(VITTransformerBlock, self).__init__()
        self.cfg = cfg

        self.C = self.cfg.hidden_features_size
        self.attention = VITSelfAttention(cfg)
        self.features = nn.Sequential(nn.Linear(self.C, self.C), get_activation(self.cfg.transformer_activation))
        self.adaptive_norm = nn.Sequential(nn.Linear(self.C, 2*self.C), get_activation("sigmoid"))
        self.norm = nn.LayerNorm(self.C)
    
    def _normalize(self, tokens: SequenceTensor) -> SequenceTensor:
        (scale, shift) = self.adaptive_norm(tokens).view(*tokens.shape[:-1], self.C, 2).unbind(dim=-1)
        normalized_tokens = scale * tokens + shift
        return normalized_tokens
    
    def forward(self, input: SequenceTensor) -> VITBlockOutput:

        tokens_normalized = self.norm(input)
        attention = self.attention(tokens_normalized)
        tokens_att = (input + attention)

        tokens_normalized = self._normalize(tokens_att)
        tokens = self.features(tokens_normalized)
        tokens = (tokens + tokens_att)
        
        return VITBlockOutput(tokens, 
                              tokens_normalized, 
                              tokens_att)
        
        


class VITTransformer(nn.Module):
    def __init__(self, cfg: VITTransformerConfig) -> None:
        super(VITTransformer, self).__init__()
        self.cfg = cfg

        (Pw, Ph) = (self.cfg.image_size[0] // self.cfg.patch_size[0], 
                    self.cfg.image_size[1] // self.cfg.patch_size[1])
        self.patches_n = (Pw, Ph)

        self.embeddings = PatchEmbedder(cfg)
        self.blocks = nn.ModuleList([
            VITTransformerBlock(cfg)
            for _ in range(self.cfg.num_transformer_blocks)
        ])
    
    def forward(self, input: ImageTensor) -> SequenceTensor:
        
        x = self.embeddings(input)
        hidden_activations = []
        attention_activations = []
        for idx, block in enumerate(self.blocks):
            block_output = block(x)
            x = block_output.features
            if idx in self.cfg.out_hidden_indices:
                hidden_activations.append(x)
                attention_activations.append(block_output.attention)
        
        return VITTransformerOutput(self.patches_n,
                                x,
                                hidden_activations,
                                attention_activations)
                
        

# if __name__ == "__main__":

#     cfg = _VIT_CONFIGURATIONS["tiny"]
#     model = VITTransformer(cfg)
#     test = torch.normal(0, 1, (10, 1, *cfg.image_size))
    
#     output = model(test)
#     spatial_tensors = output.get_spatial_tokens
#     print(output.last_activation.size(), spatial_tensors.last_activation.size())
    


        
        

