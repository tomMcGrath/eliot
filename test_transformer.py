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
D_MLP = 4 * D_MODEL


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


class TestSoftmax:

    def test_build_causal_mask(self):
        mask = transformer.build_causal_mask(T)
        torch.testing.assert_close(
            mask.triu(diagonal=1),
            torch.full(mask.size(), -torch.inf).triu(diagonal=1)
        )
        torch.testing.assert_close(
            mask.tril(),
            torch.zeros(mask.size()).tril()
        )

    def test_apply_causal_mask(self):
        logits = torch.randn(size=(B, N_HEADS, T, T))
        mask = transformer.build_causal_mask(T)
        weights = transformer.masked_softmax(logits, mask)
        torch.testing.assert_close(
            weights.triu(diagonal=1),
            torch.zeros_like(weights).triu(diagonal=1),
        )

    def test_final_dim_sums_to_one(self):
        logits = torch.randn(size=(B, N_HEADS, T, T))
        mask = transformer.build_causal_mask(T)
        weights = transformer.masked_softmax(logits, mask)
        torch.testing.assert_close(
            torch.sum(weights, dim=-1),
            torch.ones(size=(B, N_HEADS, T)),
            )


class TestWeightedSum:

    def test_is_identity(self):
        v = torch.randn(size=(B, T, N_HEADS, D_HEAD))
        weights = torch.eye(T).repeat(B, N_HEADS, 1, 1)
        identity_outputs = transformer.weighted_sum(weights, v)
        torch.testing.assert_close(identity_outputs, v)


class TestAttention:

    def test_shape(self):
        attn = transformer.Attention(N_HEADS, D_HEAD, D_MODEL, T)
        input = torch.randn(size=(B, T, D_MODEL))
        output = attn(input)
        torch.testing.assert_close(
            input.shape,
            output.shape
        )


class TestTransformerBlock:

    def test_shape(self):
        layer = transformer.TransformerBlock(
            D_MODEL, N_HEADS, D_HEAD, T, D_MLP)
        input = torch.randn(size=(B, T, D_MODEL))
        output = layer(input)
        torch.testing.assert_close(
            input.shape,
            output.shape
        )
