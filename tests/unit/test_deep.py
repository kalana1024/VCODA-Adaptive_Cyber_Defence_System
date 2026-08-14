import pytest

torch = pytest.importorskip("torch")
from vcoda.models.deep import build_network


@pytest.mark.parametrize("architecture,shape", [
    ("mlp", (4, 12)),
    ("cnn1d", (4, 16, 12)),
    ("cnn_lstm", (4, 16, 12)),
    ("transformer", (4, 16, 12)),
    ("autoencoder", (4, 12)),
])
def test_all_deep_architectures_forward(architecture, shape):
    model = build_network(architecture, input_dim=12, classes=2, profile="small", sequence_length=16)
    output = model(torch.randn(*shape))
    assert output.shape[0] == 4
    assert output.shape[-1] == (12 if architecture == "autoencoder" else 2)


def test_focal_loss_is_finite():
    from vcoda.models.deep import FocalLoss

    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.2]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = FocalLoss(alpha=torch.tensor([1.0, 2.0]), gamma=2.0)(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
