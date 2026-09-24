import pytest
import torch
from torch import nn

from leafnet.model import HybridClassifier, build_model, count_parameters


class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


def test_hybrid_concatenates_branch_features():
    model = HybridClassifier({"cnn": (Pool(6), 6), "vit": (Pool(5), 5)}, num_classes=4)
    images = torch.randn(2, 3, 16, 16)
    assert model.feature_dim == 11
    assert model.features(images).shape == (2, 11)
    assert model(images).shape == (2, 4)


def test_gradients_reach_every_branch():
    model = HybridClassifier({"cnn": (Pool(6), 6), "vit": (Pool(5), 5)}, num_classes=3)
    loss = nn.functional.cross_entropy(model(torch.randn(4, 3, 8, 8)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    for branch in model.branches.values():
        assert branch.conv.weight.grad is not None
        assert branch.conv.weight.grad.abs().sum() > 0


def test_branch_with_wrong_dimension_fails_loudly():
    model = HybridClassifier({"cnn": (Pool(6), 7)}, num_classes=3)
    with pytest.raises(RuntimeError, match="expected"):
        model(torch.randn(1, 3, 8, 8))


@pytest.mark.parametrize("kwargs", [{"branches": {}, "num_classes": 3}, {"num_classes": 1}])
def test_constructor_validation(kwargs):
    kwargs.setdefault("branches", {"cnn": (Pool(2), 2)})
    with pytest.raises(ValueError):
        HybridClassifier(**kwargs)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("resnet9000", 10, pretrained=False)


def test_reported_architecture_dimensions():
    """ResNet-101 (2048-D) + ViT-B/16 (768-D) fuse into the 2816-D vector in the write-up."""
    model = build_model("hybrid", num_classes=100, pretrained=False)
    assert model.branch_dims == {"cnn": 2048, "vit": 768}
    assert model.feature_dim == 2816
    assert model.head.in_features == 2816 and model.head.out_features == 100
    with torch.no_grad():
        assert model.eval()(torch.randn(1, 3, 224, 224)).shape == (1, 100)
    # 42.5M (ResNet-101 trunk) + 85.8M (ViT-B/16) + 0.28M (head)
    assert 128_000_000 < count_parameters(model) < 129_000_000


def test_single_branch_ablations_share_the_head_design():
    vit_only = build_model("vit", num_classes=5, pretrained=False)
    assert list(vit_only.branches) == ["vit"]
    assert vit_only.head.in_features == 768
