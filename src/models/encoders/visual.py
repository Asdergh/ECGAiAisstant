import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from .registry import (
    register_encoder, 
    get_activation,  
    register_config
)
from ..types import *


_IN_CHANNELS = {
    "image": 3,
    "signal": 1
}

class VITBlockOutput(NamedTuple):
    features: PatchTensor
    film_norm: PatchTensor
    attention: PatchTensor


@register_config("vit-visual-encoder")
@dataclass
class VITTransformerConfig:
    name: Optional[str]="vit"
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
    def __init__(
        self, 
        activation_fn: Optional[str]="relu",
        hiden_features_size: Optional[int]=None,
        attention_pool_scale: Optional[int]=None,
        patches_n: Optional[Tuple[int, int]]=None,
        cfg: Optional[VITTransformerConfig]=None
    ) -> None:
        super(VITSelfAttention, self).__init__()
        self.cfg = cfg
        
        if self.cfg is not None:
            self.C = self.cfg.hidden_features_size
            activation_fn = self.cfg.attention_activation
            self.attention_pool = self.cfg.attention_pool_scale
            self.patches_n = (self.cfg.image_size[0] // self.cfg.patch_size[0], 
                              self.cfg.image_size[1] // self.cfg.patch_size[1])
        else:
            self.C = hiden_features_size
            self.attention_pool = attention_pool_scale
            self.patches_n = patches_n

        self.d_scale = (math.sqrt(self.C)) ** -1
        self.quary = nn.Sequential(nn.Linear(self.C, self.C), get_activation(activation_fn))
        self.key = nn.Sequential(nn.Linear(self.C, self.C), get_activation(activation_fn))
        self.value = nn.Sequential(nn.Linear(self.C, self.C), get_activation(activation_fn))

    def _sequence2spatial(self, tokens: SequenceTensor) -> PatchTensor:
        B = tokens.size(0)
        tokens = tokens.transpose(-1, -2)
        spatial_tokens = tokens.view(B, self.C, *self.patches_n)
        return spatial_tokens

    def _spatial2sequence(self, tokens: PatchTensor) -> SequenceTensor:
        sequence_tokens = torch.flatten(tokens, start_dim=-2)
        sequence_tokens = sequence_tokens.permute(0, 2, 1)
        return sequence_tokens
    
    def forward(
        self, quary: SequenceTensor, 
        key: Optional[SequenceTensor]=None,
        value: Optional[SequenceTensor]=None
    ) -> SequenceTensor:
        
        key = (quary if key is None else key)
        value = (key if value is None else value)
        Q, K, V = (
            self._sequence2spatial(self.quary(quary)), 
            self._sequence2spatial(self.key(key)), 
            self._sequence2spatial(self.value(value))
        )
        if (self.attention_pool is not None) \
            and (self.patches_n is not None):
            q = F.avg_pool2d(Q, (self.attention_pool, 1))
            k = F.avg_pool2d(K, (self.attention_pool, 1))
            
            Qk = self.d_scale * torch.einsum("ncik, nckj -> ncij", Q, k.transpose(-1, -2))
            Qk = F.softmax(Qk, dim=-1)
           
            Kq = self.d_scale * torch.einsum("ncik, nckj -> ncij", q, K.transpose(-1, -2))
            Kq = F.softmax(Kq, dim=-1)
            
            Kqv = torch.einsum("ncik, nckj -> ncij", Kq, V)
            QkKqv = torch.einsum("ncik, nckj -> ncij", Qk, Kqv)
            return self._spatial2sequence(QkKqv)

        else:
            QK = self.d_scale * torch.einsum("nik, nkj -> nij", Q, K)
            weights = F.softmax(QK, dim=-1)
            QKV = torch.einsum("nik, nkj -> nij", weights, V)
            return self._spatial2sequence(QKV)




class VITTransformerBlock(nn.Module):
    def __init__(self, cfg: VITTransformerConfig) -> None:
        super(VITTransformerBlock, self).__init__()
        self.cfg = cfg

        self.C = self.cfg.hidden_features_size
        self.attention = VITSelfAttention(cfg=cfg)
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
        
        

@register_encoder("vit")
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
                
        

print(type(VITTransformerConfig))
# if __name__ == "__main__":

#     cfg = _VIT_CONFIGURATIONS["tiny"]
#     model = VITTransformer(cfg)
#     test = torch.normal(0, 1, (10, 1, *cfg.image_size))
    
#     output = model(test)
#     spatial_tensors = output.get_spatial_tokens
#     print(output.last_activation.size(), spatial_tensors.last_activation.size())
    


        
        

