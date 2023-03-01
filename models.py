import torch
import transformer


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


class RNNModel(torch.nn.Module):
    """The venerable RNN."""

    def __init__(self, vocab_size, embedding_size, num_layers, dtype=torch.float32):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dtype = dtype
        self._setup_model()

    def _setup_model(self):
        self._embedder = Embedding(
            self.vocab_size, self.embedding_size, self.dtype)
        self._rnn = torch.nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=self.num_layers,
            batch_first=True,
            )
        self._unembedder = torch.nn.Linear(
            self.embedding_size, self.vocab_size, self.dtype)

    def forward(self, tokens):
        embeds = self._embedder(tokens)
        rnn_out, hiddens = self._rnn(embeds)
        logits = self._unembedder(rnn_out)
        return logits, hiddens


class TransformerModel(torch.nn.Module):
    """A conventional Transformer language model."""

    def __init__(self, vocab_size, num_layers, d_model, n_heads, d_head, maxlen, d_mlp):
        super().__init__()
        self._embedder = Embedding(vocab_size, d_model)
        self._pos_embeddings = Embedding(maxlen, d_model)
        self._unembedder = torch.nn.Linear(d_model, vocab_size)
        transformer_blocks = [
            transformer.TransformerBlock(d_model, n_heads, d_head, maxlen, d_mlp)
            for _ in range(num_layers)]
        self._net = torch.nn.Sequential(*transformer_blocks)

    def forward(self, tokens):
        _, seq_len = tokens.shape
        token_positions = torch.arange(
            seq_len, device=tokens.device).view(1, seq_len)
        z = self._embedder(tokens) + self._pos_embeddings(token_positions)
        z = self._net(z)
        logits = self._unembedder(z)
        return logits
