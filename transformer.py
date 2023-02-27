import torch
import numpy as np


def compute_attn_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Computes attention logits for query and key embeddings."""
    b, t, n_heads, d_head = q.shape
    q = q.view((b, n_heads, t, d_head))  # [B, H, T, d_head]
    k_T = k.permute(0, 2, 3, 1)  # [B, H, d_head, T]
    attn_logits = (q @ k_T) / np.sqrt(d_head)  # [B, H, T, T]
    return attn_logits


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applies an additive mask and softmaxes across final dim."""
    return torch.nn.functional.softmax(logits + mask, dim=-1)
