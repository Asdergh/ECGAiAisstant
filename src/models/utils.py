import cv2
import torch as th
import h5py as h5
import numpy as np
import tqdm as tq
import pandas as pd
import wfdb as wf
import os 

from scipy.signal import stft
from pywt import cwt
from torchvision.transforms import (
    Resize,
    functional
)
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


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

    if len(tensor.size()) == 2:
        out = []
        for signal_idx in range(tensor.size()[0]):
            
            cwt_tensor, _ = cwt(
                tensor[signal_idx, :spec_size[-1]].numpy(), 
                np.arange(1, spec_size[0] + 1), 
                wavelet="morl"
            )
            out.append(th.Tensor(cwt_tensor).unsqueeze(dim=0))
        
        out = th.cat(out, dim=0)
    
    else:

        cwt_tensor, _ = cwt(
            tensor[:spec_size[-1]].numpy(),
            np.arange(1, spec_size[0] + 1), 
            wavelet="morl"
        )
        out = th.Tensor(cwt_tensor).unsqueeze(dim=0)
    
    return out


def collate_ImgLabels(batch):

    images = []
    labels = []
    for (img, label) in batch:
        img = _vision_tfs_["resize"](functional.pil_to_tensor(img)).to(th.float32)
        img = img / img.max()
        images.append(img.unsqueeze(dim=0))
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



def signals2waves(paths: dict, ) -> None:

    root = paths["root"]
    target_path = paths["target_path"]
    annots = pd.read_csv(paths["annotations"])

    for idx in tq.tqdm(
        range(len(annots["EGREFID"])),
        colour="red",
        ascii=":>",
        desc="[ Converting Data ]"
    ):
        
        sample = annots.iloc[idx, :]
        out_path = os.path.join(target_path, f"sample{idx}.pt")
        wave = convert_to_wavelet(wf.rdrecord(os.path.join(
            root,
            str(sample["RANDID"]),
            sample["EGREFID"]
        )).p_signal)

        th.save((
            wave,
            sample["RR"],
            sample["PR"], 
            sample["QT"], 
            sample["QRS"]
        ), out_path)


def extract_text_from_df(
    path: str, target_path: str, 
    cols, sep: str = ".",
) -> None:

    texts = ""
    df = pd.read_csv(path)
    for idx in tqdm(
        range(df.shape[0]),
        colour="green",
        ascii="=>-"
    ):
        
        string = ""
        for report in cols:

            sample = df.iloc[idx, :][report]
            string += f"{sample}{sep}"
        

        texts += f"{string}\n"
    
    with open(target_path, "w") as file:
        file.write(texts)




    

if __name__ == "__main__":

    path = "C:\\Users\\1\\Downloads\\test_data_ECG.csv"
    target_path = "C:\\Users\\1\\Desktop\\PythonProjects\\ECG_project\\meta\\token_generator_meta\\features.txt"
    # extract_text_from_df(
    #     path=path,
    #     target_path=target_path,
    #     cols=["rr_interval", "pr_interal", "qt_interval", "qrs_onset"],
    #     sep=" "
    # )
    # path = "C:\\Users\\1\\Downloads\\test_data_ECG.csv"
    # df = pd.read_csv(path)
    # df["pr_interal"] = df["qrs_onset"] - df["p_end"] 
    # df["qt_interval"] = df["t_end"] - df["qrs_end"]
    # df.to_csv(path)
    test = np.loadtxt(target_path)
    print(test.shape)

    



    

