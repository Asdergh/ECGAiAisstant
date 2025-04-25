import torch as th
import numpy as np
import os

from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from attention_text_generator import AttEmbnedding




class ConvAttTrainer:

    def __init__(
        self,
        model: Module,
        optim: Optimizer,
        criterion: Module,
        loader: DataLoader,
        target_path: str
    ) -> None:

        self.model = model
        self._optim_ = optim
        self._criterion_ = criterion
        self._loader_ = loader

        self.train_history = {}
        self._train_sess_ = 0
        self._target_path_ = target_path

    def _save_weights_(self) -> None:

        weights_d = os.path.join(
            self._target_path_,
            "weights"
        )
        if not os.path.exists(weights_d):
            os.mkdir(weights_d)
    
        num_versions = len(os.listdir(weights_d)) + 1
        th.save(self.model.state_dict(), os.path.join(
                weights_d, f"model_weights_{num_versions}.pt"
            ))

    def train(self, epochs: int, batch_per_save: int = 1, save_samples: bool = True) -> None:
        
        losses = []
        for idx in range(epochs):
            
            local_loss = 0.0
            for idx, (inputs, labels) in enumerate(tqdm(
                self._loader_,
                colour="blue",
                ascii="=->",
                desc=f"Epizode: [{idx}]"
            )):

                self._optim_.zero_grad()
                
                if th.isnan(labels).any():
                    continue

                target = self.model(inputs)
                loss = self._criterion_(target, labels)
                loss.backward()
                local_loss +=  loss.item()

                self._optim_.step()
                if (idx % batch_per_save) == 0:
                    self.model.backbone.show_activations(
                        inputs[0, :, :, :].unsqueeze(dim=0),
                        save=save_samples,
                        path=os.path.join(self._target_path_, "activations")
                    )

            losses.append(local_loss)
        
        self.train_history[self._train_sess_] = np.asarray(losses)
        self._train_sess_ += 1
        self._save_weights_()
        

class LLMReinforceTrainer:


    def __init__(
        self,
        llm: Module,
        optimizer: Optimizer,
        loader: DataLoader,
        tokenizer: AutoTokenizer,
        embedder: Module
    ) -> None:
        
        self.llm = llm
        self._embedder_ = embedder
        self._optim_ = optimizer
        self._loader_ = loader
        self.tokenizer = tokenizer

        self.train_history = {}
        self._session_number_ = 0

    def train(self, epochs: int) -> None:

        losses = []
        for idx in tqdm(
            range(epochs),
            colour="green",
            ascii="=>-",
            desc=f"Traning Session: [{self._session_number_}]"
        ):
            
            local_loss = 0.0
            for (features, texts) in tqdm(
                self._loader_,
                colour="red",
                ascii="=>-",
                desc=f"Epoch: [{idx}]"
            ):
                
                self._optim_.zero_grad()
                inputs = self.tokenizer(texts, return_tensors="pt")
                embedding = self._embedder_(inputs, features)
                output = self.model(inputs_embeds=embedding, labels=texts)
                
                
                
                loss = output.loss
                local_loss += loss.item()
                loss.backward()

                self._optim_.step()
            
            losses.append(local_loss)
        
        self.train_history[f"epizode_{self._session_number_}"] = {
            "losses": np.asarray(losses) 
        }
        self._session_number_ += 1

                

class GPTModelForceTrainer:

    def __init__(
        self,
        gpt: Module,
        optimizer: Optimizer,
        loader: DataLoader,
        traning_seq_length: int=30,
        device: str="cpu"
    ) -> None:
        
        self.gpt = gpt
        self.optim = optimizer
        self.loader = loader
        self.train_seq_len = traning_seq_length
        self.device = device

        self.train_history = {}
        self._train_session_ = 0
    
    def train(self, epochs: int, stop_batch: int=None) -> None:

        losses = []
        for _ in range(epochs):

            local_loss = 0.0
            for idx, (feature_vec, texts_ids) in enumerate(tqdm(
                self.loader,
                colour="red",
            )):

                self.optim.zero_grad()

                texts = self.loader.dataset.tokenizer.sequences_to_texts(texts_ids)
                loss = self.gpt(feature_vec, texts)
                loss.backward()
                local_loss += loss.item()

                self.optim.step()
                if stop_batch is not None:
                    if stop_batch == idx:
                        break
            
            losses.append(local_loss)
        
        self.train_history[f"{self._train_session_}"] = np.asarray(losses)
        self._train_session_ += 1
            
            

                
                
                