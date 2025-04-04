import cv2
import torch as th
import h5py as h5
import numpy as np
from scipy.signal import stft
from pywt import cwt

from torchvision.transforms import (
    Resize,
    functional
)
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10



_vision_tfs_ = {
    "resize": Resize((128, 128)),
}
def convert_to_spec(tensor: np.ndarray, spec_size: tuple = (129, 129)) -> th.Tensor:

    if len(tensor.shape) == 2:
        out = []
        for signal_idx in range(tensor.shape[-1]):
            
            _, _, Sxx = stft(tensor[:, signal_idx])
            Sxx = cv2.resize(Sxx.real, spec_size)
            out.append(th.Tensor(Sxx).unsqueeze(dim=0))
    
    else:
        _, _, out = stft(tensor, fs=10e7, nperseg=100, scaling="psd")
        out = cv2.resize(out.real, spec_size)
        
    
    return out


def convert_to_wavelet(tensor: th.Tensor, spec_size: tuple = (128, 128)):

    if len(tensor.shape) == 2:
        out = []
        for signal_idx in range(tensor.shape[-1]):
            
            cwt_tensor, _ = cwt(tensor[:spec_size[0], signal_idx], np.arange(1, spec_size[-1] + 1), wavelet="morl")
            # cwt_tensor = cwt_tensor[:, :spec_size[0]]
            out.append(th.Tensor(cwt_tensor).unsqueeze(dim=0))
        
        out = th.cat(out, dim=0)
    
    else:

        cwt_tensor = cwt(tensor.numpy())
        out = th.Tensor(cwt_tensor)
    
    return out


def collate_ImgLabels(batch) -> tuple[th.Tensor]:

    images = []
    labels = []
    for (img, label) in batch:
        img = _vision_tfs_["resize"](size=(256, 256))(functional.pil_to_tensor(img))
        images.append(img)
        labels.append(th.tensor(label).unsqueeze(dim=0))
    
    return (th.cat(images, dim=0), th.cat(labels, dim=0))
    
def build_cifar10(
    path: str, 
    train: bool = True, 
    batch_size: int = 32
):

    dataset = CIFAR10(
        root=path,
        download=False,
        train=train
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_ImgLabels
    )

    return loader

