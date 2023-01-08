import torch

class Embedding(torch.nn.Module):
    """Embeds token indices of shape [B, T]."""
    
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
        # tokens = torch.tensor(tokens)  # consider moving conversion to torch upstream?
        token_one_hots = torch.nn.functional.one_hot(
            tokens, num_classes=self.vocab_size).to(self.dtype)
        return self.embed(token_one_hots)


class LinearModel(torch.nn.Module):
    """It's an LLM! (linear language model)"""

    def __init__(self, vocab_size, embedding_size, dtype=torch.float32):
        super(LinearModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dtype = dtype
        self._setup_model()

    def _setup_model(self):
        self._embedder = Embedding(
            self.vocab_size, self.embedding_size, self.dtype)
        self._unembedder = torch.nn.Linear(
            self.embedding_size, self.vocab_size, self.dtype)
        self._model = torch.nn.Sequential(self._embedder, self._unembedder)

    def forward(self, tokens):
        logits = self._model(tokens)
        return logits
