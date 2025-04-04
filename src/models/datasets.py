import torch as th
import tqdm as tq
import h5py as h5
import wfdb as wf
import pandas as pd
import os



from torch.utils.data import Dataset, DataLoader
from utils import convert_to_spec, convert_to_wavelet


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
        spec =  self._signal_transform_[self.params["tf_type"]](sample, spec_size=self.params["spec_size"])
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
        self.annots = self.annots[self.annots["RANDID"] <= 2004]
    

    def __len__(self) -> int:
        return len(self.annots["EGREFID"].to_list())

    def __getitem__(self, idx: int) -> tuple:

        sample = self.annots.iloc[idx, :]
        signal = self._signal_transform_["wavelet"](wf.rdrecord(os.path.join(
            self.params["root"],
            str(sample["RANDID"]),
            sample["EGREFID"]
        )).p_signal)

        return (signal, sample["RR"], sample["PR"], sample["QT"], sample["QRS"])
        
        
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


 


    
        
    


if __name__ == '__main__':
    # Имитация данных ЭКГ с параметрами QRS
    ecg_samples = [
        {
            'rr_interval': 0.8,
            'p_onset': 0.1,
            'p_end': 0.2,
            'qrs_onset': 0.3,
            'qrs_end': 0.4,
            't_end': 0.6,
            'p_axis': 45.0,
            'qrs_axis': 60.0,
            't_axis': 70.0,
        },
        {
            'rr_interval': 0.9,
            'p_onset': 0.15,
            'p_end': 0.25,
            'qrs_onset': 0.35,
            'qrs_end': 0.45,
            't_end': 0.65,
            'p_axis': 50.0,
            'qrs_axis': 65.0,
            't_axis': 75.0,
        }
    ]

    # Соответствующие отчеты
    reports = [
        "Отчет для примера 1: параметры в норме",
        "Отчет для примера 2: небольшое отклонение",
    ]

    dataset = ECGDataset(ecg_samples, reports)
    sample = dataset[0]
    print("ECG данные:", sample["ecg"])
    print("Отчет:", sample["report"])