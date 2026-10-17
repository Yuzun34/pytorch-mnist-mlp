from pathlib import Path

import torch
from PIL import Image, ImageDraw

from mnist_mlp.handwriting import preprocess_handwritten_image, segment_student_number_image


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


def test_segment_student_number_image_extracts_all_11_digits():
    student_img = Path(__file__).resolve().parents[1] / "data" / "handwritten_digits" / "number.png"
    if student_img.exists():
        segments = segment_student_number_image(str(student_img))
        assert len(segments) == 11
        for seg in segments:
            assert seg["mnist28_pil"].size == (28, 28)
            assert seg["mnist28_pil"].mode == "L"
