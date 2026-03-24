import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn.modules.sparse import Embedding
import math

tokenizer = AutoTokenizer.from_pretrained("/Users/mithunnayak/Desktop/WORK/gpt-2/gpt_tokeniser")
# tokenizer.save_pretraied("./gpt_tokeniser") #downloading the tokeniser
vocabulary_size = tokenizer.vocab_size
# print(tokenizer("hello may name is mayur")['input_ids'])

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:, :x.size(1)].shape)
        return x + self.pe[:, :x.size(1)]

class GPT_Transformer(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.pos_encod = PositionalEncoding(d_model=64, max_len=100)
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embeddings = Embedding(vocab_size, model_dim)

    def forward(self, x):
        print(f"Input string: {x}")
        tokens = torch.tensor(tokenizer(x)['input_ids']).unsqueeze(0)
        print(f"Tokens {tokens}")
        print(f"Tokens shape {tokens.shape}")
        embed = self.embeddings(tokens)
        print(f"Embeddings shape {embed.shape}")
        positional_encoding_embeddings = self.pos_encod(embed)
        print(f"Positional Encoding Embeddings shape is {positional_encoding_embeddings.shape}")
        
        return positional_encoding_embeddings

pos_encod = PositionalEncoding(d_model=64, max_len=100)
model = GPT_Transformer(model_dim=64, vocab_size=vocabulary_size)
model("hi my name is mayur")


