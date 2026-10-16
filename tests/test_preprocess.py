from pathlib import Path

import torch
from PIL import Image, ImageDraw

from mnist_mlp.handwriting import preprocess_handwritten_image


def test_preprocess_handwritten_image_outputs_mnist_compatible_tensor(tmp_path: Path):
    image_path = tmp_path / "digit.png"
    image = Image.new("L", (80, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), "7", fill=0)
    image.save(image_path)

    tensor = preprocess_handwritten_image(str(image_path))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 28, 28)
    assert torch.isfinite(tensor).all()
