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


_activations_ = {
    "relu": ReLU,
    "softmax": Softmax,
    "sigmoid": Sigmoid
}
class TokenGenerator(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.emb_dim = params["embedding_dim"]
        self.vocab_size = params["vocab_size"]
        self.hiden_size = params["hiden_size"]
        self.lstm_heads = params["lstm_heads"]
        self.num_att_heads = params["num_att_heads"]

        
        self.seq_len = None
        if "seq_len" in params:
            self.seq_len = params["seq_len"]

        self.input_dim = None
        if "input_dim" in params["input_dim"]:
            self.input_dim = params["input_dim"]

        _gen_type_ = params["gen_type"]
        _builders_ = {
            "toke_gen": self._build_token_gen_,
            "seq_gen": self._build_seq_gen_
        }
        _builders_[_gen_type_]()

        assert (_gen_type_ == "seq_gen") and (self.seq_len is None & self.input_dim is None), f"""

        If you want to use seq generator insted of next token prdictor you must 
        specify the size of input encoded vector and seq_lenn. Curent values:
        
            -> seq_len: [{self.seq_len}] <-
            -> input_size: [{self.input_dim}] <-

        """
        

        self.dp_rate = 0.0
        if "dp_rate" in params:
            self.dp_rate = params["dp_rate"]

        
    
    
    # def _build_seq_gen_(self) -> None:


    #     self._projection_ = Sequential(
    #         Linear(self.input_dim, self.seq_len),
    #         _activations_["relu"]
    #         Embedding(
                
    #         )
    #     )
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
        self._rnn_ = LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hiden_size,
            num_layers=self.lstm_heads
        )
        self._attention_ = MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_att_heads
        )

    def __call__(self, inputs: th.Tensor, reinforce: th.Tensor = None) -> th.Tensor:

        emb = self._embedding_(inputs)
        if reinforce is not None:
            
            weights = th.normal(0.0, 1.0, (reinforce.size()[-1], self.emb_dim)).T
            reinforce = functional.linear(reinforce, weights)
            reinforce = reinforce.view(1, reinforce.size()[0], self.emb_dim)
        
            att_embedding = self._attention_(emb.permute(1, 0, 2), reinforce, reinforce)[0].permute(1, 0, 2)
            emb += att_embedding
        
        rnn_out, _ = self._rnn_(emb)
        rec_out = th.flatten(
            rnn_out, 
            start_dim=1, 
            end_dim=-1
        )
        weights = th.normal(0.0, 1.0, (rec_out.size()[-1], self.hiden_size)).T
        rec_out = functional.linear(rec_out, weights)
        
        return self._logits_(rec_out)




        


if __name__ == "__main__":

    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    VOCAB_SIZE = 10000
    HIDEN_SIZE = 100
    LSTM_HEADS = 8
    ATT_HEADS = 8
    SEQ_LEN = 64

    test_input = th.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    reinforce = th.normal(0.0, 1.1, (BATCH_SIZE, 100))
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
        
    