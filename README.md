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