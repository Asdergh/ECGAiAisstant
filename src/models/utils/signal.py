import numpy as np
import torch
import torch.nn.functional as F
import neurokit2 as nkt2
import matplotlib.pyplot as plt

from wfdb import rdrecord
from scipy import signal
from ..types import *


def apply_stft2signals(
    signals: Union[np.ndarray, torch.Tensor], 
    nperseg: int=1000, 
    noverlap: int=300,
    target_size: Tuple[int, int]=(112, 448),
    return_tensors: str="pt",
    max_freqs_n: Optional[int]=25
):
    if isinstance(signals, np.ndarray):
        signals = torch.from_numpy(signals)
    spectrograms = torch.stft(
        signals,
        n_fft=nperseg,
        hop_length=(nperseg - noverlap),
        win_length=noverlap,
        window=torch.hann_window(noverlap),
        center=True,
        normalized=True,
        onesided=True,
        return_complex=True
    )    
    spectrograms = (torch.abs(spectrograms))[:, None, :max_freqs_n, :]
    spectrograms = F.interpolate(spectrograms, target_size).squeeze()
    return (spectrograms if return_tensors == "pt" else spectrograms.numpy())

def read_signal(path: str) -> np.ndarray:

    records = rdrecord(path)
    (signals, sample_rate) = (records.p_signal, records.fs)
    signals = signals.T
    scale = 1 / (sample_rate * 100)

    RRs = []; PRs = []
    QRSs = []; QTs = []
    signals_cleaned = []
    for idx in range(signals.shape[0]):

        signal_cleaned = nkt2.ecg_clean(signals[idx, :], sample_rate)
        signals_cleaned.append(signal_cleaned)
        (_, waves) = nkt2.ecg_delineate(signal_cleaned)
        intervals = {
            "RR": [], "QRS": [],
            "QT": [], "PR": [] 
        }
        num_peaks = min([len(values) for values in waves.values()])
        for idx in range(num_peaks - 1):
            
            pr = (waves["ECG_Q_Peaks"][idx] - waves["ECG_P_Onsets"][idx]) * scale
            qrs = (waves["ECG_S_Peaks"][idx] - waves["ECG_Q_Peaks"][idx]) * scale 
            qt = (waves["ECG_T_Offsets"][idx] - waves["ECG_Q_Peaks"][idx]) * scale
            rr = (waves["ECG_R_Onsets"][idx + 1] - waves["ECG_R_Onsets"][idx]) * scale
            intervals['RR'].append(rr)
            intervals["PR"].append(pr)
            intervals["QRS"].append(qrs)
            intervals["QT"].append(qt)
        
        RRs.append(np.mean(np.asarray(intervals["RR"])))
        PRs.append(np.mean(np.asarray(intervals["PR"])))
        QTs.append(np.mean(np.asarray(intervals["QT"])))
        QRSs.append(np.mean(np.asarray(intervals["QRS"])))
    
    signals_cleaned = np.stack(signals_cleaned, axis=0)
    PRs = np.asarray(PRs); RRs = np.asarray(RRs)
    QTs = np.asarray(QTs); QRSs = np.asarray(QRSs)
    return {
        "signals": signals,
        "signals_cleaned": signals_cleaned,
        "RR": RRs, "PR": PRs,
        "QT": QTs, "QRS": QRSs
    }




if __name__ == "__main__":

    import time
    import os
    from torchvision.utils import make_grid

    s_time = time.time()
    path = "/home/ram/Downloads/ptb-diagnostic-ecg-database-1.0.0/patient001/s0010_re"

    params = [
        {"nperseg": 900, "noverlap": 220, "max_freqs_n": 10},
        {"nperseg": 2500, "noverlap": 1020, "max_freqs_n": 10},
        {"nperseg": 1000, "noverlap": 320, "max_freqs_n": 10}
    ]
    specs = []
    signal_pkg = read_signal(path)
    # signals = (signal_pkg["signals"] + 1) / 2
    signals = signal_pkg["signals"]
    for kwargs in params:
        sample_idx = np.random.randint(0, 15, (3, ))
        spectrograms = apply_stft2signals(signals, **kwargs)
        spectrograms = spectrograms[None, sample_idx, ...].transpose(0, 1)
        spectrograms = make_grid(spectrograms, nrow=3).mean(dim=0)[None]
        print(spectrograms.size())
        specs.append(spectrograms)
    
    specs = torch.stack(specs, dim=0)
    specs = make_grid(specs, nrow=1).mean(dim=0)

    e_time = time.time()
    _, axis = plt.subplots()
    axis.imshow(specs)
    plt.show()
    