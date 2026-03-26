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
        self.W_o = torch.nn.Parameter(torch.rand(d_v, d_in))

    def forward(self, x):
        B, T, _ = x.shape

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        d_head_q = self.d_q // self.heads
        d_head_v = self.d_v // self.heads

        Q = q.view(B, T, self.heads, d_head_q).transpose(1, 2)  # (B, H, T, d_head_q)
        K = k.view(B, T, self.heads, d_head_q).transpose(1, 2)  # (B, H, T, d_head_q)
        V = v.view(B, T, self.heads, d_head_v).transpose(1, 2)  # (B, H, T, d_head_v)

        scores = Q @ K.transpose(-2, -1)                        # (B, H, T, T)
        scores = scores / (d_head_q ** 0.5)                     
        attn = f.softmax(scores, dim=-1)                       

        values = attn @ V                                      
        values = values.transpose(1, 2).contiguous()           
        values = values.view(B, T, self.d_v)                   

        out = values @ self.W_o                                
        return out

    

# model = MHA(8,8,8,2)
# model(embedded_sentence.unsqueeze(0))



class MHA_KV_CACHE(nn.Module):
    def __init__(self, d_in, d_q, d_v, d_heads):
        super().__init__()
        self.d_q = d_q
        self.d_v = d_v
        self.heads = d_heads

        self.w_q = nn.Parameter(torch.rand(d_in, d_q))
        self.w_k = nn.Parameter(torch.rand(d_in, d_q))
        self.w_v = nn.Parameter(torch.rand(d_in, d_v))
        self.W_o = nn.Parameter(torch.rand(d_v, d_in))

    def forward(self, x, kv_full=None):
        B, T, _ = x.shape

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        # split into heads
        d_head_q = self.d_q // self.heads
        d_head_v = self.d_v // self.heads

        Q = q.view(B, T, self.heads, d_head_q).transpose(1, 2)
        K = k.view(B, T, self.heads, d_head_q).transpose(1, 2)
        V = v.view(B, T, self.heads, d_head_v).transpose(1, 2)

        # KV cache
        if kv_full is not None:
            past_K, past_V = kv_full
            K = torch.cat((past_K, K), dim=-2)
            V = torch.cat((past_V, V), dim=-2)

        present_kv = (K, V)

        # attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (d_head_q ** 0.5)
        attn = f.softmax(scores, dim=-1)

        values = attn @ V

        # merge heads
        values = values.transpose(1, 2).contiguous()
        values = values.view(B, T, self.d_v)

        out = values @ self.W_o

        return out, present_kv





import torch
import time

def benchmark_mha(mha, mha_kv, seq_len=128, d_in=64, device="cpu"):
    mha = mha
    mha_kv = mha_kv

    x_full = torch.randn(1, seq_len, d_in)

    print(f"\nSequence Length: {seq_len}")

    # ---------------------------
    # 1. Full MHA (no KV cache)
    # ---------------------------
    start = time.time()

    for t in range(1, seq_len + 1):
        x = x_full[:, :t, :]          
        _ = mha(x)

    full_time = time.time() - start
    print(f"Full MHA Time: {full_time:.6f} sec")

    # ---------------------------
    # 2. KV Cache MHA
    # ---------------------------
    start = time.time()

    kv_cache = None
    for t in range(seq_len):
        x = x_full[:, t:t+1, :]       # one token at a time
        _, kv_cache = mha_kv(x, kv_cache)

    kv_time = time.time() - start
    print(f"KV Cache Time: {kv_time:.6f} sec")

    # ---------------------------
    # Speedup
    # ---------------------------
    print(f"Speedup: {full_time / kv_time:.2f}x")



d_in = 1280
d_q = 1280
d_v = 1280
heads = 20

mha = MHA(d_in, d_q, d_v, heads)
mha_kv = MHA_KV_CACHE(d_in, d_q, d_v, heads)

benchmark_mha(mha, mha_kv, seq_len=128, d_in=d_in)