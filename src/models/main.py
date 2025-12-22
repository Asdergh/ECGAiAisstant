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


def normalize_wave_input(wave: th.Tensor, target_channels: int = 12) -> th.Tensor:
    """
    Приводит wave к форме [B, target_channels, H, W]
    Поддерживает входы:
      - [C, H, W]
      - [B, C, H, W]
    """

    # --- приводим к 4D ---
    if wave.dim() == 3:
        wave = wave.unsqueeze(0)  # [1, C, H, W]

    if wave.dim() != 4:
        raise ValueError(f"Unsupported wave shape: {wave.shape}")

    b, c, h, w = wave.shape

    # --- если каналов ровно сколько надо ---
    if c == target_channels:
        return wave

    # --- если 1 канал → размножаем ---
    if c == 1:
        return wave.repeat(1, target_channels, 1, 1)

    # --- если каналов меньше ---
    if c < target_channels:
        repeats = (target_channels + c - 1) // c
        wave = wave.repeat(1, repeats, 1, 1)
        return wave[:, :target_channels]

    # --- если каналов больше ---
    if c > target_channels:
        return wave[:, :target_channels]

    return wave


class PipeLine:

    def __init__(
            self,
            config: Union[str, dict, List[str]],
            weights: Union[str, OrderedDict] = None,
            local_storage: str = "meta_local"
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

        # FIX: режим inference — отключаем training state
        self._conv_att_net_.eval()

        # FIX: НЕ храним историю сообщений в объекте
        self._system_message_ = None
        if "system_message" in self.config:
            self._system_message_ = SystemMessage(
                content=self.config["system_message"]
            )

        self._giga_ = GigaChat(
            credentials=GIGACHAT_AUT_KEY,
            verify_ssl_certs=False,
        )

        print("MODEL!!")
        print(self._conv_att_net_)

    def _parse_ecg(
            self,
            path: str,
            records_n: int = 3,
            sampling_rate: float = 1e-4
    ) -> None:

        ecg_path = os.path.join(self.local_storage, "ecg_records")
        if not os.path.exists(ecg_path):
            os.mkdir(ecg_path)

        # FIX: корректная очистка каталога
        for f in os.listdir(ecg_path):
            os.remove(os.path.join(ecg_path, f))

        zip.ZipFile(path).extractall(ecg_path)

        for content in os.listdir(ecg_path):
            fnt = content.split(".")
            if fnt[-1] in ["hea", "dat"]:
                fpath = os.path.join(ecg_path, fnt[0])
                break

        record = wf.rdrecord(fpath)
        signals = record.p_signal.T
        del record  # FIX: освобождаем тяжёлый объект

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

        # FIX: ОБЯЗАТЕЛЬНО закрываем figure
        plt.close(fig)

    def __call__(self, path: str) -> str:
        print("STARTED")

        record = wf.rdrecord(path)
        signal = record.p_signal
        del record  # FIX: освобождаем память

        signal = th.tensor(signal).T
        print(signal.size())

        wave = convert_to_wavelet(signal)

        wave = normalize_wave_input(wave, target_channels=12)

        print(wave.size())

        # FIX: отключаем autograd — КРИТИЧНО для long-running сервиса
        with th.no_grad():
            features = (self._conv_att_net_(wave) * 100).to(th.int)

        print(features.size())

        content = (
            "Дай описание показателям ЭКГ с параметрами RR, PR, "
            "QT, QRS, имеющими следующие значения: "
            f"{features[0].tolist()}. "
            "Опиши как можно точнее."
        )

        # FIX: создаём сообщения НА КАЖДЫЙ ВЫЗОВ, без очереди
        messages = []
        if self._system_message_ is not None:
            messages.append(self._system_message_)

        messages.append(HumanMessage(content=content))

        inv = self._giga_.invoke(messages)
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
    

        

        
        
        
        
        

        
        
        

        

