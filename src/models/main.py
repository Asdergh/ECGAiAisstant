import torch as th 
import os 
import yaml, json
import wfdb as wf
import requests 
from collections import OrderedDict


from typing import Union, BinaryIO, IO, List

from conv_attention_model import *
from config import *
from utils import *
from pprint import pprint


_read_io_ = {
    "yaml": yaml.safe_load,
    "json": json.load
}


class PipeLine:

    def __init__(
        self, 
        config: Union[str, dict, List[str]],
        weights: Union[str, OrderedDict]=None
    ) -> None:

        self.config = config
        if isinstance(config, str):

            if os.path.isfile(config):
                _, file_type = os.path.split(config)
                with open(config, "r") as file:
                    self.config = _read_io_[file_type](file)
            
            else:
                self.config = {}
                for file in os.listdir(config):
                    file = os.path.join(config, file)
                    _, file_type = os.path.split(file)
                    with open(file, "r") as file:
                        config = _read_io_[file_type](file)
                        self.config.update(config)

        if isinstance(config, list):

            self.config = {}
            for path in config:

                config_type = os.path.basename(path)
                _, f_type = os.path.split(path)
                assert config_type in ["signal_analytic", "generation_config", "gen_details"], f"""
                Is you wan't to pass a List of files in format List[str], you must
                ensure that your config files names have the same baenames as the
                config types names curently allowed: 

                    - signal_analytic: file with configurations for signal analisation model
                    - generation_config: file with configuration for generation models, in 
                        usual case with standard parameters for tokens generation models
                    - gen_details: file with additional details for LLM in text format
                All this files can be ether in .yaml or .json types
                """
                with open(path, "r") as file:
                    content = _read_io_[f_type](file)
                    config = {config_type: content}
                    self.config.update(config)
                
        self._signal_net_ = Model(self.config["signal_analytic"])
        if weights is not None:
            if isinstance(weights, OrderedDict):
                self._signal_net_.load_state_dict(weights)
            else:
                weights = th.load(weights, weights_only=True)
                self._signal_net_.load_state_dict(weights)
        
        else:
            path = "C:\\Users\\1\\Desktop\\PythonProjects\\ECG_project\\meta\\conv_att_training_path\\weights\\model_weights_1.pt"
            weights = th.load(path, weights_only=True)
            self._signal_net_.load_state_dict(weights)

        print(type(self._signal_net_.state_dict()))
        self._payload_ = {}
        if "generation_config" in self.config:
            self._payload_.update({"parameters": self.config["generation_config"]})
    
    @property
    def generation_config(self) -> dict:
        if "parameters" in self._payload_:
            return self._payload_["parameters"]
        return None

    @generation_config.setter
    def generation_config(self, config: dict) -> None:
        self._payload_["parameters"] = config

    def __call__(self, path: str, MISTRAL_URL=None) -> str:
        print("STARTED")
        signal = wf.rdrecord(path).p_signal
        signal = th.Tensor(signal).T
        print(signal.size()) 
        wave = convert_to_wavelet(signal)
        print(wave.size())

        features = (self._signal_net_(wave) * 100).to(th.int)
        print(features.size())
        prompt = f"""
            Дай описание показателям экг с параметрами RR, PR,
            QT, QRS, имеющими след значения: {features[0].tolist()}
        """
        
        if "gen_details" in self.config:
            for (key, value) in self.config["gen_details"].items():
                prompt = f"""
                    {key.upper()}:
                        {value}
                    {(32 * '-')}"
                    {prompt}
                """
        
        self._payload_.update({
            "inputs": prompt
        })
        print(self._payload_["inputs"])
        return requests.post(
            MISTRAL_URL,
            headers=HEADERS,
            json=self._payload_
        ).json()
    

if __name__ == "__main__":

    config = {
        "signal_analytic": {
            "backbone": {
                "projection": {
                    "in_channels":   [12, 32, 64  ],
                    "out_channels":  [32, 64, 128 ],
                    "embedding_dim": [32, 32, 64  ]
                },
                "conv": {
                    "kernel_size": (3, 3),
                    "padding": 1,
                    "stride": 2
                },
                "pool": {
                    "kernel_size": (3, 3),
                    "padding": 1,
                    "stride": 1
                }
            },
            "out_head": {
                "type": "multyoutput",
                "params": {
                    "embedding_dim": 16,
                    "dims": [4]
                }
            }
        },
        "generation_config": {
            "max_new_tokens": 200,         
            "temperature": 0.7,             
            "top_p": 0.9,                  
            "repetition_penalty": 1.1,      
            "do_sample": True,
            "num_return_sequences": 1
        },
        "gen_details": {
            "роль": "Ты помощник медицинского эксперта",
            "задача": """
                Ты должен генерировать промежуточные диагнозы на 
                основании диагнозов параметров экг таких как 
                RR, PR, QT, QRS, которые ты будешь получать 
                в формате текста но в виде list.
            """,
            "результат": """
                Текст с полным описанием экг пациента по 
                RR, PR, QT, QRS, дополнительными рекомендациями
                от себя, и предупрежедением о том что ты лишь 
                ИИ ассистент и пользователю стоит обратиться за 
                более точной консультацией к медработнику. 
                Все поля раздели между собой линиями.
            """
        }
    }

    path = "test/00689D31-8491-4643-B3C8-45241FBBD47C"
    pipeline = PipeLine(config=config, weights="meta/model_weights_1.pt")
    out = pipeline(path, MISTRAL_URL=MISTRAL_URL)
    print(out)
    

        

        
        
        
        
        

        
        
        

        

