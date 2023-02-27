import numpy as np
import torch
import transformer


# Define test attributes
# TODO: consider moving to parameterized testing
T = 3
B = 5
N_HEADS = 7
D_HEAD = 11
D_MODEL = N_HEADS * D_HEAD

class TestLogits:

    def test_shape_is_correct(self):
        """Check that the shape of logits is correct."""
        q = torch.randn(size=(B, T, N_HEADS, D_HEAD))
        k = torch.randn(size=(B, T, N_HEADS, D_HEAD))
        logits = transformer.compute_attn_logits(q, k)
        assert logits.shape == (B, N_HEADS, T, T)

    def test_is_identity(self):
        """Check that we get identity attention logits given suitable inputs."""
        # Single head, single timestep, normalised for pre-softmax factor
        q = torch.eye(T, D_HEAD).view((1, T, 1, D_HEAD)) * np.sqrt(D_HEAD)
        k = torch.eye(T, D_HEAD).view((1, T, 1, D_HEAD))
        logits = transformer.compute_attn_logits(q, k)
        torch.testing.assert_close(logits[0, 0], torch.eye(T, T))

    def test_uniform(self):
        """Check that we get uniform attention logits given suitable inputs."""
        # Normalise q for pre-softmax factor sqrt(d_head) and sum over d_head
        q = torch.ones((B, T, N_HEADS, D_HEAD)) * np.sqrt(D_HEAD) / D_HEAD
        k = torch.ones((B, T, N_HEADS, D_HEAD))
        logits = transformer.compute_attn_logits(q, k)
        torch.testing.assert_close(logits, torch.ones((B, N_HEADS, T, T)))