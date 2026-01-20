import torch as th
import tqdm as tq
import h5py as h5
import wfdb as wf
import pandas as pd
import os
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset, DataLoader
from models.add_utils import convert_to_spec, convert_to_wavelet


class SignalsSet(Dataset):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params
        self._signals_ = h5.File(self.params["path"], "r")["tracings"]
        self._signal_transform_ = {
            "spectrogram": convert_to_spec,
            "wavelet": convert_to_wavelet
        }
    
    def __len__(self) -> int:
        return self._signals_.shape[0]

    def __getitem__(self, idx: int) -> None:
        
        sample = th.Tensor(self._signals_[idx])
        spec =  self._signal_transform_[self.params["tf_type"]](
            sample, 
            spec_size=self.params["spec_size"]
        )
        return spec



class ECGSpecsDataset(Dataset): 
    
    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params
        self._signal_transform_ = {
            "spectrogram": convert_to_spec,
            "wavelet": convert_to_wavelet
        }
        self.annots = pd.read_csv(self.params["annotations"])
    

    def __len__(self) -> int:
        return len(self.annots["EGREFID"].to_list())

    def __getitem__(self, idx: int) -> tuple:

        sample = self.annots.iloc[idx, :]
        signal_ = self._signal_transform_["wavelet"](wf.rdrecord(os.path.join(
            self.params["root"],
            str(sample["RANDID"]),
            sample["EGREFID"]
        )).p_signal).to(th.float32)
        signal_ = signal_ / signal_.max()

        features_ = th.Tensor([sample["RR"], sample["PR"], sample["QT"], sample["QRS"]]).to(th.float32) / (100.0)
        return (signal_, features_)
        
        
class ECGDataset(Dataset):
    def __init__(self, ecg_data, reports, transform=None):
        """
        Инициализация датасета.

        Аргументы:
            ecg_data (list): Список записей с данными ЭКГ. Каждая запись должна быть словарем с параметрами:
                'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'
            reports (list): Список отчетов (строк), соответствующих каждой записи ЭКГ.
            transform (callable, опционально): Функция для преобразования данных ЭКГ.
        """
        assert len(ecg_data) == len(reports), "Количество данных ЭКГ должно совпадать с количеством отчетов."
        self.ecg_data = ecg_data
        self.reports = reports
        self.transform = transform

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_sample = self.ecg_data[idx]
        report = self.reports[idx]

        # Если задано преобразование, применяем его
        if self.transform:
            ecg_sample = self.transform(ecg_sample)

        # Преобразуем значения параметров в тензор
        keys = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
        ecg_tensor = th.tensor([ecg_sample[key] for key in keys], dtype=th.float)

        return {"ecg": ecg_tensor, "report": report}   
    

class MedicalReportsSet(Dataset):


    def __init__(self, params: dict) -> None:

        super().__init__()
        
        self._texts_ = params["texts"]
        self._features_ = params["features"]
        self.trunc_depth = params["trunc_depth"]
        
        self.return_obj = "idxs"
        if "return_type" in params:
            self.return_obj = params["return_type"]
        
          
        if isinstance(self._texts_, str):
            with open(self._texts_, "r") as file:
                
                self._texts_ = file.readlines()
                max_len = float("inf")
                for sample in self._texts_:
                    sample = sample.split(" ")
                    if max_len < len(sample) + 2:
                        max_len = len(sample) + 2

                for idx, sample in enumerate(self._texts_):   
                
                    sample = sample.replace("\\n", "").replace("\\t", " ").split(" ")
                    sample = self._truncation_(trunc_depth=self.trunc_depth, texts=sample)
                    sample = ["<BOS>", ] + sample + ["<EOS>", ]
                    self._texts_[idx] = sample

        if isinstance(self._features_, str):
            self._features_ = np.loadtxt(self._features_)

        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self._texts_)
        self._sequences_ = np.asarray(self.tokenizer.texts_to_sequences(self._texts_))
        

        _idxs_ = [
            idx for idx in range(self._features_.shape[0]) 
            if (self._features_[idx] < 0).any()
        ]
        self._texts_ = [text for (idx, text) in enumerate(self._texts_) if idx not in _idxs_]
        self._features_ = np.delete(self._features_, _idxs_, axis=0)
        self._sequences_ = np.delete(self._sequences_, _idxs_, axis=0)
        

    def _truncation_(self, trunc_depth: int, texts: list) -> list:
        
        if len(texts) < trunc_depth:
            texts += ["<PAD>" for _ in range(trunc_depth - len(texts))]
        
        else:
            texts = texts[:trunc_depth]
         
        return texts


    def __len__(self) -> None:
        return len(self._sequences_)


    def __getitem__(self, idx: int) -> None:
        
        if self.return_obj == "idxs":
            return (
                th.Tensor(self._features_[idx] / 100.0).to(th.float32), 
                th.Tensor(self._sequences_[idx]).to(th.int32)
            )
    
        elif self.return_obj == "texts":
            return (
                th.Tensor(self._features_[idx] / 100.0).to(th.float32),
                self._texts_[idx]
            )
    
    

        
if __name__ == "__main__":
    
    # dataset = MedicalReportsSet({
    #     "texts": "C:\\\\Users\\\\1\\\\Desktop\\\\PythonProjects\\\\ECG_project\\\\meta\\\\token_generator_meta\\\\texts.txt",
    #     "collate_fn": {
    #         "type": "truncate",
    #         "params": {
    #             "trunc_depth": 7
    #         }
    #     }
    # })
    # print(np.asarray(dataset.tokenizer.texts_to_sequences(dataset._texts_)).shape)

    

    texts = "C:\\Users\\1\\Desktop\\PythonProjects\\ECG_project\\meta\\token_generator_meta\\texts.txt"
    features = "C:\\Users\\1\\Desktop\\PythonProjects\\ECG_project\\meta\\token_generator_meta\\features.txt"
    dataset = MedicalReportsSet({
        "texts": texts,
        "features": features,
        "trunc_depth": 7,
        "return_type": "texts"
    })
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    
    sample = next(iter(loader))
    # print(sample[0].size(), sample[1])
    

        


    
        
    