import numpy as np
import torch
import neurokit2 as nkt2
import matplotlib.pyplot as plt

from wfdb import rdrecord
from scipy import signal



def read_signal(path: str) -> np.ndarray:

    signals = rdrecord(path).p_signal
    signal_cleaned = nkt2.ecg_clean(signals[:, 0])
    (r_peaks, meta) = nkt2.ecg_peaks(signal_cleaned)
    (signals, waves) = nkt2.ecg_delineate(signal_cleaned)
    print(type(signals))

    idx = np.random.randint(0, int(signal_cleaned.shape[0] // 100))
    sample = signal_cleaned[idx * 100: (idx + 1) * 100]
    keys = list("PSQT")
    Peaks_idx = {
        key: np.asarray(
            waves[f"ECG_{key}_Peaks"][idx * 100: (idx + 1) * 100]
        )
        for key in keys
    }
    _, axis = plt.subplots()
    axis.plot(sample, color="blue")
    for key, peaks_idx in Peaks_idx.items():
        ax = axis.twinx()
        ax.vlines(
            x=peaks_idx, 
            ymin=np.ones_like(peaks_idx) * -0.4,
            ymax=np.ones_like(peaks_idx) * 0.4,
            label=f"{key} peaks",
            color=np.random.rand(3, ), 
            linestyle="--"
        )
    plt.show() 
    
    
    
    

if __name__ == "__main__":

    path = "/home/ram/Downloads/ptb-diagnostic-ecg-database-1.0.0/patient001/s0010_re"
    read_signal(path)