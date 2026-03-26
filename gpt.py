import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn.modules.sparse import Embedding
import math

tokenizer = AutoTokenizer.from_pretrained("/Users/mithunnayak/Desktop/WORK/gpt-2/gpt_tokeniser")
# tokenizer.save_pretraied("./gpt_tokeniser") #downloading the tokeniser
vocabulary_size = tokenizer.vocab_size
# ids = tokenizer("hello may name is mayur")['input_ids']
# ids_to_str = tokenizer.decode(ids)
# print(ids_to_str)


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
    def __init__(self, model_dim, vocab_size, blocks):
        super().__init__()
        self.blocks = blocks
        self.pos_encod = PositionalEncoding(d_model=model_dim, max_len=1000)
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embeddings = Embedding(vocab_size, model_dim)
        self.mha = torch.nn.MultiheadAttention(model_dim, 12,batch_first=True)
        self.linear1 = torch.nn.Linear(model_dim, model_dim * 4)
        self.linear2 = torch.nn.Linear(model_dim * 4, model_dim)
        self.relu = torch.nn.ReLU()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.llm_head = nn.Linear(self.model_dim, self.vocab_size)


    def generate_causal_mask(self,seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).float()
        return mask

    def forward(self, x):
        # print(f"Input string: {x}")
        tokens = torch.tensor(tokenizer(x)['input_ids']).unsqueeze(0)
        # print(f"Tokens {tokens}")
        # print(f"Tokens shape {tokens.shape}")
        embed = self.embeddings(tokens)
        # print(f"Embeddings shape {embed.shape}")
        positional_encoding_embeddings = self.pos_encod(embed)
        # print(f"Positional Encoding Embeddings shape is {positional_encoding_embeddings.shape}")
        x_in = positional_encoding_embeddings
        seq_len = x_in.size(1)
        mask = self.generate_causal_mask(seq_len)
        for i in range(self.blocks):
            x_in = self.norm1(x_in)
            attn_output, attn_weights  = self.mha(x_in,x_in,x_in,attn_mask=mask)
            # print(f"Self attention block shape {attn_output.shape}")
            x_in = self.norm1(x_in + attn_output)
            ff = self.linear2(self.relu(self.linear1(x_in)))
            # print(f"shape after ffn {ff.shape}")
            x_in = self.norm2(x_in + ff)
        logits = self.llm_head(x_in)
        prob = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(prob, dim=-1)
        return next_token 




model = GPT_Transformer(model_dim=768, vocab_size=vocabulary_size, blocks=12)
def generate_tokens(model, x, num_tokens=30):
        for i in range(num_tokens):
            next_t = model(x)
            next_token = next_t[0][-1]
            next_word = tokenizer.decode(next_token)
            x = x + next_word
            print(next_word)


generate_tokens(model, "Hi, my name ", 50) #text generation

total_params = sum(p.numel() for p in model.parameters())
print("Total params:", total_params)
# print(f"the final output size is: {model("hi my name is mayur").shape}")
# next_token = model("hi my name is mayur")[0][-1]
# print(next_token[0][-1])
# print(tokenizer.decode(next_token))





