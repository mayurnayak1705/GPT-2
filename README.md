Mini GPT Transformer (PyTorch)

This project implements a simplified version of a GPT-like Transformer using PyTorch. It takes a string input, tokenizes it, and generates subsequent tokens based on learned patterns.

Features
Tokenization using a custom GPT-2 tokenizer
Embedding layer for converting tokens to vector space
Positional encoding to inject token position information
Multi-head self-attention (causal mask for autoregression)
Feedforward network (FFN) inside each block
Layer normalization and residual connections
Final linear layer to map embeddings back to vocabulary space
Softmax to compute probabilities and generate the next token


Model Architecture
Embedding Dimension (model_dim): 768
Number of Transformer Blocks: 12
Number of Attention Heads: 12
Vocabulary Size: Derived from the custom tokenizer
Feedforward Layer Expansion: 4× model_dim


Notes on Generation
The generation process uses greedy decoding. For more diverse results, you can modify the generate_tokens() function to implement sampling techniques (e.g., top-k, top-p, temperature scaling).
The model is currently untrained. The output will not be meaningful until trained on a large language dataset.


KV-CHACHE
The efficiency gain from using KV cache during autoregressive generation. In the standard MHA implementation, at each step the model recomputes attention over the entire sequence built so far, leading to increasing computational cost as the sequence grows. In contrast, the KV cache version avoids recomputing past keys and values, processing only the new token at each step while reusing previously stored information. This reduces redundant computation and results in a noticeable speedup—about 2.76× in your case for sequence length 128. The improvement isn’t extremely large here because the sequence length is still moderate; as you scale to longer sequences (e.g., 256, 512, 1024), the gap will widen significantly, and KV caching will provide much larger performance benefits, which is why it is a core optimization in transformer models like GPT-2.