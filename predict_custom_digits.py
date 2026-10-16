import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from train_mnist_mlp import MLP


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(image).unsqueeze(0)


def predict(model: MLP, image_paths: List[Path], device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        for path in image_paths:
            tensor = preprocess_image(path).to(device)
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            print(f"{path.name}: {pred}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict custom handwritten digits with a trained MLP")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing digit images")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.hidden_dims) < 2:
        raise ValueError("Model must have at least two hidden layers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=28 * 28, hidden_dims=args.hidden_dims).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    image_dir = Path(args.image_dir)
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    predict(model, image_paths, device)


if __name__ == "__main__":
    main()
