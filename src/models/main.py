import torch as th 
import os 
import json
import wfdb as wf
import requests 
import zipfile as zip
import PIL
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from collections import OrderedDict
from typing import Union, List
from conv_attention_model import *
from config import *
from utils import *

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from config import GIGACHAT_AUT_KEY





class PipeLine:

    def __init__(
        self, 
        config: Union[str, dict, List[str]],
        weights: Union[str, OrderedDict]=None,
        local_storage: str="meta_local"
    ) -> None:

        self.config = config
        self.local_storage = local_storage
        if isinstance(self.config, str):
            with open(self.config, "r") as file:
                self.config = js.load(file)
                
        self._conv_att_net_ = Model(self.config["ConvAttNet"])
        if weights is not None:
            if isinstance(weights, OrderedDict):
                self._conv_att_net_.load_state_dict(weights)
            else:
                weights = th.load(weights, weights_only=True)
                self._conv_att_net_.load_state_dict(weights)
        
        else:
            path = "C:\\Users\\1\\Desktop\\PythonProjects\\ECG_project\\meta\\conv_att_training_path\\weights\\model_weights_1.pt"
            weights = th.load(path, weights_only=True)
            self._conv_att_net_.load_state_dict(weights)

        self._messages_ = []
        self._giga_ = GigaChat(
            credentials=GIGACHAT_AUT_KEY,
            verify_ssl_certs=False,
        )

        print("MODEL!!")
        print(self._conv_att_net_)
        if "system_message" in self.config:
            self._messages_.append(SystemMessage(content=self.config["system_message"]))
    

   
    def _parse_ecg(
        self, 
        path: str, 
        records_n: int=3, 
        sampling_rate: float=1e-4
    ) -> None:
        
        ecg_path = os.path.join(self.local_storage, "ecg_records")
        if not os.path.exists(ecg_path):
             os.mkdir(ecg_path)

        if os.listdir(ecg_path) != 0:
            for f in os.listdir(ecg_path):
                fpath = os.path.join(ecg_path, f)
                os.remove(fpath)
        
        zip.ZipFile(path).extractall(ecg_path)
        for content in os.listdir(ecg_path):
            fnt = content.split(".")
            if fnt[-1] in ["hea", "dat"]:
                fpath = os.path.join(ecg_path, fnt[0])
                break
        
        signals = wf.rdrecord(fpath).p_signal.T
        if records_n > signals.shape[0]:
            records_n = signals.shape[0]
        
        n_samples = int(signals.shape[-1] * sampling_rate)
        signals = signals[:records_n, :n_samples]
        fig, axis = plt.subplots(nrows=records_n)
        for idx in range(records_n):
            signal = signals[idx]
            axis[idx].plot(signal, color=np.random.rand(3), label=f"record{idx}")
            axis[idx].legend(loc="upper left")
        
        fimg = os.path.join(ecg_path, "records_plot.png")
        fig.savefig(fimg)

        
    def __call__(self, path: str) -> str:
        print("STARTED")
        signal = wf.rdrecord(path).p_signal
        signal = th.Tensor(signal).T
        print(signal.size()) 
        wave = convert_to_wavelet(signal)
        if wave.size()[1] != 12:
            wave = wave.repeat(1, 6, 1, 1)

        print(wave.size())

        features = (self._conv_att_net_(wave) * 100).to(th.int)
        print(features.size())
        content = f"""
            Дай описание показателям экг с параметрами RR, PR,
            QT, QRS, имеющими след значения: {features[0].tolist()}. 
            Oпишите как можно точнее
        """
        self._messages_.append(HumanMessage(
            content=content,
            max_token=200
        ))
        inv = self._giga_.invoke(self._messages_)
        return inv.content
        
        
    

if __name__ == "__main__":

    config = {
        "ConvAttNet": {
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
        "system_message": "Вы — помощник диагноста ЭКГ для создания временных диагнозов"
    }

    path = "/home/ramzan/Desktop/PythonProjects/ECGAiAisstant/meta/001"
    pipeline = PipeLine(config=config, weights="/home/ramzan/Desktop/PythonProjects/ECGAiAisstant/meta/model_weights_1.pt")
    out = pipeline(path)
    print(out)
    

        

        
        
        
        
        

        
        
        

        

