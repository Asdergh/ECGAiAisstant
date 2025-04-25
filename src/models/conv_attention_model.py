import torch as th
import matplotlib.pyplot as plt
import os
import json as js
plt.style.use("dark_background")


from typing import (
    Union,
    List,
    Tuple
)
from torch import flatten
from torchvision.transforms import Resize
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
    SiLU,
    AvgPool2d,
    MaxPool2d,
    functional
)
from math import log2


_activations_ = {
    "relu": ReLU,
    "tanh": Tanh,
    "softmax": Softmax,
    "sigmoid": Sigmoid,
    "silu": SiLU
}


class ConvAttentionBlock(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params

        self._in_conv_ = Conv2d(
            in_channels=self.params["in_channels"], 
            out_channels=self.params["out_channels"],
            **self.params["conv"]
        )
        self._hiden_conv_ = Conv2d(
            in_channels=self.params["out_channels"] * 2, 
            out_channels=1,
            kernel_size=(3, 3),
            padding=1,
            stride=1
        )

        self._pooling_ = ModuleDict({
            "avg": AvgPool2d(**self.params["pool"]),
            "max": MaxPool2d(**self.params["pool"])
        })
        self._pooling_embedding_ = ModuleDict({
            "avg": Linear(self.params["embedding_dim"], self.params["out_channels"]),
            "max": Linear(self.params["embedding_dim"], self.params["out_channels"]),
        })
    
        self._act_vector_ = _activations_["softmax"](dim=1)
        self._act_img_ = _activations_["softmax"](dim=1)

    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        conv = self._in_conv_(inputs)
        avg_pool_ft = flatten(
            self._pooling_["avg"](conv),
            start_dim=1,
            end_dim=-1
        )
        max_pool_ft = flatten(
            self._pooling_["max"](conv),
            start_dim=1,
            end_dim=-1
        )
        
        avg_embedding_weights = th.normal(0.0, 1.0, (
            avg_pool_ft.size()[1], 
            self.params["embedding_dim"]
        )).T
        max_embedding_weights = th.normal(0.0, 1.0, (
            max_pool_ft.size()[1], 
            self.params["embedding_dim"]
        )).T

        avg_embedding = (functional.linear(
            avg_pool_ft, 
            avg_embedding_weights
        ))
        max_embedding = (functional.linear(
            max_pool_ft, 
            max_embedding_weights
        ))
       
        avg_embedding = self._pooling_embedding_["avg"](avg_embedding)
        max_embedding = self._pooling_embedding_["max"](max_embedding)
        embedding = self._act_vector_(avg_embedding + max_embedding)
        embedding = embedding.view(
            embedding.size()[0], 
            self.params["out_channels"], 
            1, 1
        )
        hiden_conv = th.mul(conv, embedding)
        
        avg_pool = self._pooling_["avg"](hiden_conv)
        max_pool = self._pooling_["max"](hiden_conv)
        depth_map = self._act_img_(th.cat([avg_pool, max_pool], dim=1))
        depth_map = self._hiden_conv_(depth_map)

        conv = conv + depth_map
        return conv


class ConvAttentionEncoder(Module):

    def __init__(self, params: Union[dict, str]) -> None:

        super().__init__()
        self.params = params
        if isinstance(self.params, str):
            with open(self.params, "r") as json:
                self.params = js.load(json)

        self._activations_saved_ = 0
        self._conv_att_ = []
        _projections_ = self.params["projection"]
        for idx in range(len(_projections_["in_channels"])):

            specs = {
                "in_channels": _projections_["in_channels"][idx],
                "out_channels": _projections_["out_channels"][idx],
                "embedding_dim": _projections_["embedding_dim"][idx],
                "conv": self.params["conv"],
                "pool": self.params["pool"]
            }
            self._conv_att_.append(ConvAttentionBlock(specs))
        
        self._conv_att_ = ModuleList(self._conv_att_)
    

    def show_activations(
        self, 
        inputs: th.Tensor, 
        cmap="jet", 
        size: tuple = (128, 128), 
        save: bool = True,
        path: str = None
    ) -> None:

        fig, axis = plt.subplots()
        activations = []
        x = inputs
        resize = Resize(size)
        with th.no_grad():
            for layer in self._conv_att_:
                x = layer(x)
                activation = resize(x)
                activations.append(activation.mean(dim=1))
        
        activations = th.cat(activations, dim=2).permute(1, 2, 0).numpy()
        axis.imshow(activations, cmap=cmap)

        if save:

            if not os.path.exists(path):
                os.mkdir(path)
            fig.savefig(os.path.join(path, f"activations_{self._activations_saved_}"))
            self._activations_saved_ += 1

        else:
            plt.show()
        
    def save_confs(self, path: str) -> None:
        with open(path, "r") as json:
            js.dump(self.params, json)

    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        x = inputs
        for layer in self._conv_att_:
            x = layer(x)
        return x



class MultyHeadBlock(Module):

    def __init__(self, params: Union[dict, str]) -> None:

        super().__init__()
        self.params = params
        if isinstance(self.params, str):
            with open(self.params, "r") as json:
                self.params = js.load(json)
    

        if len(self.params["dims"]) > 1:
            self._linear_ = []

            for (idx, dim) in enumerate(self.params["dims"]):
                
                if "activations" in self.params:
                    act = _activations_[self.params["activations"][idx]]()
                
                else:
                    act = _activations_["relu"]()

                layer = Sequential(
                    Linear(self.params["embedding_dim"], dim),
                    LayerNorm(dim),
                    act
                )
                self._linear_.append(layer)
        
        else:
            
            if "activations" in self.params:
                act = _activations_[self.params["activations"][0]]()
                
            else:
                act = _activations_["relu"]()

            self._linear_ = Sequential(
                    Linear(self.params["embedding_dim"], self.params["dims"][0]),
                    LayerNorm(self.params["dims"][0]),
                    act
                )
    
    def save_confs(self, path: str) -> None:
        with open(path, "r") as json:
            js.dump(self.params, json)

    def __call__(self, inputs: th.Tensor) -> th.Tensor:


        conv = flatten(inputs, start_dim=1, end_dim=-1)
        linear = (functional.linear(conv, th.normal(0.0, 1.0, (
            conv.size()[-1], 
            self.params["embedding_dim"]
        )).T))

        if len(self.params["dims"]) > 1:
            return tuple([
                layer(linear)
                for layer in self._linear_
            ])

        else:
            out = self._linear_(linear)
            return out

class ClassificationHead(Module):

    def __init__(self, params: Union[dict, str]) -> None:

        super().__init__()
        self.params = params
        if isinstance(self.parasm, str):
            with open(self.params, "r") as json:
                self.params = js.load(json)
            
            
        self._net_ = Sequential(
            Linear(self.params["embedding_dim"], self.params["hiden_features"]),
            _activations_["relu"](),
            Linear(self.params["hiden_features"], self.params["hiden_features"]),
            _activations_["relu"](),
            Linear(self.params["hiden_features"], self.params["n_classes"]),
            _activations_["softmax"](dim=1),
        )
    
    def save_confs(self, path: str) -> None:
        with open(path, "r") as json:
            js.dump(self.params, json)


    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        
        conv = flatten(inputs, start_dim=1, end_dim=-1)
        linear = functional.linear(conv, th.normal(0.0, 1.0, (
            conv.size()[-1], 
            self.params["embedding_dim"]
        )).T)
        return self._net_(linear)
        


class Model(Module):

    def __init__(self, params: dict) -> None:
        
        super().__init__()
        self.params = params
        self._heads_ = {
            "classifier": ClassificationHead,
            "multyoutput": MultyHeadBlock
        }
        self.backbone = ConvAttentionEncoder(self.params["backbone"])
        self.out_head = self._heads_[self.params["out_head"]["type"]](self.params["out_head"]["params"])
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        if th.isnan(inputs).any():
            raise ValueError("there is nun in inputs tensor :D")
        return self.out_head(self.backbone(inputs))

if __name__ == "__main__":


#   [   usage example  ]
    classifier = Model({
        "encoder": {
            "projection": {
                "in_channels":   [3, 32, 64  ],
                "out_channels":  [32, 64, 128],
                "embedding_dim": [32, 32, 64 ]
            },
            "conv": {
                "kernel_size": (3, 3),
                "padding": 1,
                "stride": 1
            },
            "pool": {
                "kernel_size": (3, 3),
                "padding": 1,
                "stride": 2
            }
        },
        "out_head": {
            "type": "classifier",
            "params": {
                "embedding_dim": 128,
                "hiden_features": 64,
                "n_classes": 30
            }
        }
    })
   