import torch
import numpy as np


def compute_attn_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Computes attention logits for query and key embeddings."""
    b, t, n_heads, d_head = q.shape  # k should be the same
    q = q.view((b, n_heads, t, d_head))  # [B, H, T, d_head]
    k_T = k.permute(0, 2, 3, 1)  # [B, H, d_head, T]
    attn_logits = (q @ k_T) / np.sqrt(d_head)  # [B, H, T, T]
    return attn_logits


def build_causal_mask(maxlen):
    """Builds a causal mask where the upper-triangular elements are inf."""
    return torch.triu(-torch.inf * torch.ones(maxlen, maxlen), diagonal=1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applies an additive mask and softmaxes across final dim."""
    return torch.nn.functional.softmax(logits + mask, dim=-1)


def weighted_sum(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Computes the weighted sum of values across timesteps."""
    values = values.transpose(1, 2)  # [B, T, H, d_head] -> [B, H, T, d_head]
    return (weights @ values).transpose(1, 2)  # [B, T, H, d_head]


class Attention(torch.nn.Module):
    """Compute multi-head self-attention. TODO: try einsum version."""

    def __init__(self,
                 n_heads: int,
                 d_head: int,
                 d_model: int,
                 maxlen: int,
    ) -> None:
        super().__init__()

        # Define attributes
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.maxlen = maxlen

        # Define layers
        # TODO: check if fusing to a single Linear speeds things up
        self._q_proj = torch.nn.Linear(d_model, n_heads*d_head)
        self._k_proj = torch.nn.Linear(d_model, n_heads*d_head)
        self._v_proj = torch.nn.Linear(d_model, n_heads*d_head)
        self._output_linear = torch.nn.Linear(n_heads*d_head, d_model)

        # Setup causal mask
        self.register_buffer('mask', build_causal_mask(maxlen))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get shape info
        b, t, _ = input.shape
        
        # Compute queries, keys & values
        # View is doing [B, T, H*d_head] -> [B, T, H, d_head]
        q = self._q_proj(input).view((b, t, self.n_heads, self.d_head))
        k = self._k_proj(input).view((b, t, self.n_heads, self.d_head))

        # Compute logits & add causal mask to get weights
        attn_logits = compute_attn_logits(q, k)
        attn_weights = masked_softmax(attn_logits, self.mask[:t, :t])  # [B, H, T, T]

        # Sum value-weighted outputs
        v = self._v_proj(input).view((b, t, self.n_heads, self.d_head))
        flattened_outputs = weighted_sum(attn_weights, v).flatten(start_dim=2)
        return self._output_linear(flattened_outputs)


class MLP(torch.nn.Module):

    def __init__(self, d_mlp: int, d_model: int) -> None:
        super().__init__()
        self._input_linear = torch.nn.Linear(d_model, d_mlp)
        self._output_linear = torch.nn.Linear(d_mlp, d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self._input_linear(input)
        hidden = torch.nn.functional.relu(hidden)
        return self._output_linear(hidden)
