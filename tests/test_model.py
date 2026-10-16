import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import build_model, CLASSES  # noqa: E402


@pytest.mark.parametrize("name", ["cnn", "cnn_improved", "resnet"])
def test_forward_shape(name):
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, len(CLASSES))


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope")
