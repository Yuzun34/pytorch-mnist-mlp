from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .data import MNIST_MEAN, MNIST_STD, get_device
from .model import MLPClassifier


def preprocess_handwritten_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)

    image_array = torch.tensor(np.asarray(image, dtype=np.float32))
    if image_array.mean() > 127:
        image = ImageOps.invert(image)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    return transform(image)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MLPClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = MLPClassifier(
        input_size=config["input_size"],
        hidden_sizes=tuple(config["hidden_sizes"]),
        num_classes=config["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_images(checkpoint_path: str, image_paths: list[str]) -> dict[str, int]:
    device = get_device()
    model = load_model_from_checkpoint(checkpoint_path, device)

    predictions: dict[str, int] = {}
    with torch.no_grad():
        for image_path in image_paths:
            image_tensor = preprocess_handwritten_image(image_path).unsqueeze(0).to(device)
            logits = model(image_tensor)
            pred = int(logits.argmax(dim=1).item())
            predictions[image_path] = pred
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict digits for handwritten images using a trained MLP checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    predictions = predict_images(args.checkpoint, args.images)
    print(json.dumps(predictions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
