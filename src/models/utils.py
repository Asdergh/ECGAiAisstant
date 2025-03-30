import cv2
import torch as th
import h5py as h5
import numpy as np
from scipy.signal import spectrogram



def convert_to_spec(
    tensor: th.Tensor, 
    spec_size: tuple = (129, 129)
) -> th.Tensor:

    if len(tensor.size()) == 2:
        out = []
        for signal_idx in range(tensor.size()[-1]):

            _, _, Sxx = spectrogram(tensor[:, signal_idx].numpy())
            Sxx = cv2.resize(Sxx, spec_size)
            out.append(th.Tensor(Sxx).unsqueeze(dim=0))
        
        out = th.cat(out, dim=0)
    
    else:
        _, _, out = spectrogram(tensor.numpy())
        out = th.Tensor(cv2.resize(out, spec_size))
    
    return out



    

