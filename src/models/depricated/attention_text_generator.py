import torch as th
from torch.nn import (
    Linear,
    MultiheadAttention,
    LSTM,
    Embedding,
    Sequential,
    Module,
    ReLU,
    Softmax,
    Sigmoid,
    Dropout,
    functional
)
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

_activations_ = {
    "relu": ReLU,
    "softmax": Softmax,
    "sigmoid": Sigmoid
}


class AttEmbnedding(Module):
    
    def __init__(self, params) -> None:

        super().__init__()
        self.embedding_dim = params["embedding_dim"]
        self.dp_rate = params["dp_rate"]
        self.att_heads = params["att_heads"]
        self.vocab_size = params["vocab_size"]

        self.seq_len = 1
        if "seq_len" in params:
            self.seq_len = params["seq_len"]

        self._embedding_ = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )
        
        self._attention_ = MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.att_heads
        )
    
    def __call__(self, reinforce: th.Tensor, text_tokens: th.Tensor=None) -> th.Tensor:

        weights = th.normal(0.0, 1.0, (reinforce.size()[-1], self.embedding_dim)).T

        reinforce = functional.linear(reinforce, weight=weights).view(
            1, 
            reinforce.size()[0], 
            self.embedding_dim
        ).repeat(self.seq_len, 1, 1)
        reinforce = reinforce.contiguous()
        
        if text_tokens is not None:
            embedding = self._embedding_(text_tokens).permute(1, 0, 2)
            att_embedding = self._attention_(
                embedding, 
                reinforce, 
                reinforce
            )[0].permute(1, 0, 2).contiguous()
        
        else:
            att_embedding = self._attention_(
                reinforce, 
                reinforce, 
                reinforce
            )[0].permute(1, 0, 2).contiguous()

        
        if text_tokens is not None:
            embedding = embedding.permute(1, 0, 2).contiguous()
            embedding += att_embedding
        
        else:
            embedding = att_embedding

        return embedding

class TokenGenerator(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.emb_dim = params["embedding_dim"]
        self.vocab_size = params["vocab_size"]
        self.hiden_size = params["hiden_size"]
        self.lstm_heads = params["lstm_heads"]
        self.num_att_heads = params["num_att_heads"]
        



        self.dp_raet = 0.1
        if "dp_rate" in params:
            self.dp_rate = params["dp_rate"]

        self.seq_len = None
        if "seq_len" in params:
            self.seq_len = params["seq_len"]

        self.input_dim = None
        if "input_dim" in params:
            self.input_dim = params["input_dim"]

        _gen_type_ = "token_gen"
        if "gen_type" in params:
            _gen_type_ = params["gen_type"]

        _builders_ = {
            "token_gen": self._build_token_gen_,
            # "seq_gen": self._build_seq_gen_
        }
        _builders_[_gen_type_]()


    def _build_token_gen_(self) -> None:
        
        self._logits_ = Sequential(
            Linear(self.hiden_size, self.vocab_size),
            Dropout(p=self.dp_rate),
            _activations_["softmax"](dim=1)
        )

        self._embedding_ = Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.emb_dim
        )

        self._att_embedding_ = AttEmbnedding({
            "embedding_dim": self.emb_dim,
            "vocab_size": self.vocab_size,
            "dp_rate": self.dp_rate,
            "att_heads": self.num_att_heads
        })

        self._rnn_ = LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hiden_size,
            num_layers=self.lstm_heads
        )


    def __call__(self, inputs: th.Tensor, reinforce: th.Tensor = None) -> th.Tensor:

        
        if reinforce is not None:
            emb = self._att_embedding_(
                inputs,
                reinforce
            )
        
        else:
            emb = self._embedding_(inputs)
        
        rnn_out, _ = self._rnn_(emb)
        rec_out = th.flatten(
            rnn_out, 
            start_dim=1, 
            end_dim=-1
        )
        weights = th.normal(0.0, 1.0, (rec_out.size()[-1], self.hiden_size)).T
        rec_out = functional.linear(rec_out, weights)
        
        return self._logits_(rec_out)



class GPTBasedTokenGenerator(Module):


    def __init__(self, max_size: int, generation_config: dict) -> None:

        super().__init__()
        self.max_size = max_size
        self.gen_conf = generation_config

        self.emb = AttEmbnedding({
            "vocab_size": 40478,
            "embedding_dim": 768,
            "dp_rate": 0.45,
            "att_heads": 8,
            "seq_len": max_size
        })
        self.gpt = OpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-community/openai-gpt")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    

    def generate(self, embeddings: th.Tensor=None) -> str:

        if embeddings is None:
            start_token = self.tokenizer.encode("<bos>", return_tensors="pt")
            gen_tokens = self.gpt.generate(
                inputs=start_token, 
                **self.gen_conf
            ).squeeze(dim=0)
            
        
        else:
            logits = self.gpt(inputs_embeds=embeddings).logits
            gen_tokens = logits.argmax(dim=-1).squeeze(dim=0)

        print(gen_tokens.size())
        # return self.tokenizer.decode(
        #     gen_tokens, 
        #     skip_special_tokens=True
        # )
     
    def add_tokens(self, tokens: list[str]) -> None:

        self.tokenizer.add_tokens(tokens)
        vocab_size = len(self.tokenizer.get_vocab())
        self.gpt.resize_token_embeddings(vocab_size)

    def _tokenize_(self, texts: list[str]) -> th.Tensor:

        tokenization = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True
        )["input_ids"]
        
        seq_len = tokenization.size()[1]
        batch_size = tokenization.size()[0]
        if seq_len < self.max_size:
            need_size = self.max_size - seq_len
            tokenization = th.cat([
                tokenization, 
                th.zeros(batch_size, need_size)
            ], dim=1)
        
        return tokenization.to(th.long)

    def __call__(self, inputs: th.Tensor, texts: list[str]) -> th.tensor:

        tokenization = self._tokenize_(texts=texts)
        emb = self.emb(inputs)
        loss = self.gpt(inputs_embeds=emb, labels=tokenization).loss
        return loss
        
        
        




        


if __name__ == "__main__":

    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    VOCAB_SIZE = 10000
    HIDEN_SIZE = 100
    LSTM_HEADS = 8
    ATT_HEADS = 8
    SEQ_LEN = 64

    test_input = th.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    reinforce = th.normal(0.0, 1.1, (BATCH_SIZE, 4))
    model = TokenGenerator({
        "embedding_dim": EMBEDDING_DIM,
        "vocab_size": VOCAB_SIZE,
        "hiden_size": HIDEN_SIZE,
        "lstm_heads": LSTM_HEADS,
        "num_att_heads": ATT_HEADS,
        "dp_rate": 0.45
    })
    
    out = model(test_input, reinforce)
    print(out.size(), out.min(), out.max(), test_input)
        
    