import models
import torch


vocab_size = 3
num_layers = 2
d_model = 16
n_heads = 4
d_head = int(d_model / n_heads)
maxlen = 4
d_mlp = int(4 * d_model)
batch_size = 2


def setup_model():
    """Setup simple model to test."""
    model = models.TransformerModel(
        vocab_size,
        num_layers,
        d_model,
        n_heads,
        d_head,
        maxlen,
        d_mlp,
    )
    return model


def setup_data():
    """Setup data to memorise."""
    datum_to_memorise = [0, 1, 2, 0]
    memorise_me = torch.tensor(datum_to_memorise).tile(batch_size, 1)
    targets = torch.roll(memorise_me, -1, dims=1)
    return memorise_me, targets


class TestMemorisation:

    def test_can_run(self):
        """Test the model runs properly."""
        model = setup_model()
        memorise_me, _ = setup_data()
        logits = model(memorise_me)
        assert logits.shape == torch.Size([batch_size, maxlen, vocab_size])

    def test_compute_loss(self):
        """Test we can compute loss."""
        model = setup_model()
        memorise_me, targets = setup_data()
        logits = model(memorise_me)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.transpose(1, 2), targets)
        assert loss

    def test_can_diff(self):
        """Test we can backprop."""
        model = setup_model()
        memorise_me, targets = setup_data()
        logits = model(memorise_me)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.transpose(1, 2), targets)
        loss.backward()

    def test_can_memorise(self):
        """Test we can memorise a single datum."""
        model = setup_model()
        memorise_me, targets = setup_data()
        loss_fn = torch.nn.CrossEntropyLoss()
        lr = 1e-2
        optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        n_steps = int(1e3)
        for _ in range(n_steps):
            # Model prediction & loss
            logits = model(memorise_me)
            # cross-entropy expects [B, classes, ...]
            loss = loss_fn(logits.transpose(1, 2), targets)
            # Apply optimiser
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        assert loss.item() < 0.01
        torch.testing.assert_close(torch.argmax(logits, dim=-1), targets)
