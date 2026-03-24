import torch
import torch.nn as nn
import torch.nn.functional as f


vocab_size = 50_000


sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}

# print(dc)

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)
# print(sentence_int)


torch.manual_seed(123)
embed = torch.nn.Embedding(vocab_size, 8)
embedded_sentence = embed(sentence_int).detach()

# print(embedded_sentence)
print(f"shape of embedding sentence {embedded_sentence.shape}")

# d = embedded_sentence.shape[1]
# # print(d)

# d_q, d_k, d_v = 2,2,4

# W_q = torch.nn.Parameter(torch.rand(d, d_q))
# W_k = torch.nn.Parameter(torch.rand(d, d_k))
# W_v = torch.nn.Parameter(torch.rand(d, d_v))

# print(W_q.shape)
# print(W_k.shape)
# print(W_v.shape)

# q = embedded_sentence @ W_q
# k = embedded_sentence @ W_k
# v = embedded_sentence @ W_v

# print(f"shapes of q, k ,v are {q.shape} , {k.shape} , {v.shape}")


# attention_weights = q @ k.T
# print(attention_weights.shape)


# scaled_attention = f.softmax(attention_weights / d_k ** 0.5 , dim=0) #dim=0 means along the rows

# values = scaled_attention @ v

# # print(values.shape)



class MHA(nn.Module):
    def __init__(self, d_in, d_q, d_v, d_heads):
        super().__init__()
        self.d_in = d_in
        self.d_q = d_q
        self.d_v = d_v
        self.heads = d_heads

        self.w_q = torch.nn.Parameter(torch.rand(d_in, d_q))
        self.w_k = torch.nn.Parameter(torch.rand(d_in, d_q))
        self.w_v = torch.nn.Parameter(torch.rand(d_in, d_v))
        self.W_o = torch.nn.Parameter(torch.rand(d_in, d_v))

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]
        print(seq_len)
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        print(x.shape)
        print(self.w_k.shape)
        print(k.shape)


        Q = q.view(batch,seq_len, self.heads, self.d_q//self.heads).transpose(1,2)
        K = k.view(batch,seq_len, self.heads, self.d_q//self.heads).transpose(1,2)
        V = v.view(batch,seq_len, self.heads, self.d_v//self.heads).transpose(1,2)
        print(K.shape)
        attention_score = Q @ K.transpose(-2,-1)
        print(attention_score.shape)
        scaled_attention = f.softmax(attention_score / self.d_q ** 0.5 , dim=0)
        values = scaled_attention @ V
        print(values.shape)
        values = values.transpose(1,2).contiguous()
        print(values.shape)
        values = values.view(batch, seq_len, self.d_v)
        print(values.shape)
        out = values @ self.W_o
        print(out.shape)

    
# 
model = MHA(8,8,8,2)
model(embedded_sentence.unsqueeze(0))


