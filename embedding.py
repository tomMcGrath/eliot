import torch

class Embedding(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_size, dtype=torch.float32):
        """Initialises the embedding."""
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dtype = dtype
        self.embed = torch.nn.Linear(
            vocab_size, embedding_size, dtype=self.dtype)
        
        
    def forward(self, tokens):
        """Apply the embedding to a series of tokens of shape [B, T]."""
        tokens = torch.tensor(tokens)  # consider moving conversion to torch upstream?
        token_one_hots = torch.nn.functional.one_hot(
            tokens, num_classes=self.vocab_size).to(self.dtype)
        return self.embed(token_one_hots)