import torch as th
from torch import flatten
import time as t
from torch.nn import (
    Linear,
    Conv2d,
    ConvTranspose2d,
    ReLU,
    Tanh,
    Softmax,
    BatchNorm2d,
    Module,
    Sequential,
    Sigmoid,
    LayerNorm,
    Dropout,
    Flatten,
    ModuleDict,
    ModuleList,
    SiLU
)
from math import log2


_activations_ = {
    "relu": ReLU,
    "tanh": Tanh,
    "softmax": Softmax,
    "sigmoid": Sigmoid,
    "silu": SiLU
}


class ConvBlock(Module):

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: tuple = (2, 2), 
        padding: int = 0,
        stride: int = 2,
        activation: str = "relu",
        mode: str = "down"
    ) -> None:
        
        super().__init__()
        _conv_ = {
            "down": Conv2d(
                in_channels, out_channels, 
                kernel_size, stride, 
                padding
            ),
            "up": ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size, stride, 
                padding
            )
        }
        self._net_ = Sequential(
            _conv_[mode],
            BatchNorm2d(num_features=out_channels),
            _activations_[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    

class MLP(Module):

    def __init__(
        self,
        in_features: int, 
        out_features: int,
        dp_rate: float = 0.45, 
        activation: str = "relu"
    ) -> None:

        super().__init__()
        self._net_ = Sequential(
            Linear(in_features, out_features),
            LayerNorm(out_features),
            Dropout(p=dp_rate),
            _activations_[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    
class Attention(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self._projections_ = ModuleDict({
            "q": MLP(
                params["in_features"], 
                params["qk_dim"], 
                activation=params["activation"]
            ),
            "k": MLP(
                params["in_features"], 
                params["qk_dim"], 
                activation=params["activation"]
            ),
            "v": MLP(
                params["in_features"], 
                params["v_dim"], 
                activation=params["activation"]
            ),
        })
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        
        if len(inputs.size()) > 2:
            inputs = flatten(inputs, start_dim=1, end_dim=-2)
            

        q = self._projections_["q"](inputs)
        k = self._projections_["k"](inputs)
        v = self._projections_["v"](inputs)

        if len(inputs.size()) <= 2:
            probs = _activations_["softmax"](dim=1)(th.mm(q, k.T))
            att = th.mm(probs, v)
        
        else:
        
            att = []
            for idx in range(inputs.size()[1]):

                prob = th.mm(q[:, idx, :], k[:, idx, :].T)
                att_sample = th.mm(prob, v[:, idx, :]).unsqueeze(dim=1)
                att.append(att_sample)
            
            att = th.cat(att, dim=1)
            
        return att


class MultyHeadAtt(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params
        att_confs = self.params["att"]
        att_confs["in_features"] = self.params["in_features"]

        self._lin_ = MLP(
            att_confs["v_dim"] * self.params["n_heads"], 
            self.params["out_features"]
        )
        self._atts_ = [
            Attention(att_confs)
            for _ in range(self.params["n_heads"])
        ]
        self._atts_ = ModuleList(self._atts_)
    
    
    def predict(self, inputs: th.Tensor, grad: bool = False) -> th.Tensor:
        
        if grad:
            with th.no_grad():
                return self(inputs).view((
                    inputs.size()[0],
                    self.params["patch_size"][0],
                    self.params["patch_size"][1],
                    self.params["out_features"],
                ))
        
        else:
            return self(inputs).view((
                inputs.size()[0],
                self.params["patch_size"][0],
                self.params["patch_size"][1],
                self.params["out_features"],
            ))
        
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        atts = []
        for layer in self._atts_:
            att = layer(inputs)
            atts.append(att)
        
        cat = th.cat(atts, dim=-1)
        lin = self._lin_(cat)
        return lin

class Encoder(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params

        self._flatten_ = Flatten()
        _conv_params_ = self.params["conv"]
        self._conv_ = []
        for idx in range(len(_conv_params_["in_channels"])):
            conv = ConvBlock(**{
                conf: value[idx] 
                for (conf, value) in _conv_params_.items()
            })
            self._conv_.append(conv)
        
        self._conv_ = ModuleList(self._conv_)
        self._attention_ = MultyHeadAtt(self.params["matt"])

        _matt_confs_ = self.params["matt"]
        self._out_ = MLP(
            _matt_confs_["patch_size"][0] * _matt_confs_["patch_size"][1] * _matt_confs_["out_features"],
            self.params["out_features"],
            dp_rate=0.56
        )
        

    def predict(self, inputs: th.Tensor) -> th.Tensor:

        with th.no_grad():
            x = inputs
            for layer in self._conv_:            
                x = layer(x)

            return self._attention_.predict(x.permute(0, 2, 3, 1), grad=True)
        
        
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        x = inputs
        for layer in self._conv_:            
            x = layer(x)

        att = self._attention_.predict(x.permute(0, 2, 3, 1), grad=True)
        att = th.flatten(att, start_dim=1, end_dim=-1)
        return self._out_(att)


class ClssificationHead(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self._
class Decoder(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params 
        self._lin_ = MLP(
            self.params["in_features"], 
            self.params["patch_size"][0] * self.params["patch_size"][1] * self.params["conv"]["in_channels"][0]
        )

        self._conv_ = []
        _conv_params_ = self.params["conv"]
        for idx in range(len(_conv_params_["in_channels"])):
            conv = ConvBlock(**{
                conf: value[idx] 
                for (conf, value) in _conv_params_.items()
            })
            self._conv_.append(conv)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        x = self._lin_(inputs).view((
            inputs.size()[0],
            self.params["conv"]["in_channels"][0],
            self.params["patch_size"][0],
            self.params["patch_size"][1],
        ))

        for layer in self._conv_:
            x = layer(x)
        
        return x
    
        
        

class AeEncoder(Module):
    
    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params
        self.encoder = Encoder(self.params["encoder"])
        self.decoder = Decoder(self.params["decoder"])
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        return self.decoder(self.encoder(inputs))

if __name__ == "__main__":

    test = th.normal(0.12, 1.12, (10, 3, 256, 256))
    model = AeEncoder({
        "decoder": {
            "in_features": 32,
            "patch_size": (32, 32),
            "start_channels": 32,
            "conv": {
                "in_channels": [32, 64, 128],
                "out_channels": [64, 128, 3],
                "mode": ["up", "up", "up"],
                "activation": ["silu", "silu", "tanh"]
            }
        },
        "encoder": {
            "out_features": 32,
            "conv": {
                "in_channels": [3, 32, 64],
                "out_channels": [32, 64, 128],
            },
            "matt": {
                "n_heads": 10,
                "in_features": 128, 
                "out_features": 64,
                "patch_size": (32, 32),
                "att": {
                    "activation": "relu",
                    "qk_dim": 32,
                    "v_dim": 128
                }
            }
        }
    })
    


    