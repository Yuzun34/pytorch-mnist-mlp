from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from .data import MNIST_MEAN, MNIST_STD, get_device
from .model import MLPClassifier


def preprocess_pil_to_mnist28(image: Image.Image) -> Image.Image:
    """Convert a single-digit PIL image to an MNIST-style 28x28 grayscale image (black bg, white digit)."""
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # Determine if background is light (paper/white canvas) or dark (already MNIST-like)
    bg_reference = float(np.percentile(arr, 75))
    if arr.mean() > 127.0 or bg_reference > 127.0:
        # Suppress paper background and uneven lighting / faint bleed-through
        # For synthetic clean white images (bg=255, min=0), margin still preserves full stroke
        dynamic_range = bg_reference - float(arr.min())
        margin = min(22.0, max(5.0, 0.18 * dynamic_range))
        fg = np.clip((bg_reference - margin) - arr, 0.0, None)
    else:
        noise_floor = float(np.percentile(arr, 25))
        fg = np.clip(arr - noise_floor, 0.0, None)

    max_val = float(fg.max())
    if max_val > 0.0:
        fg = (fg / max_val) * 255.0

    # Find tight bounding box of the digit stroke
    threshold = 25.0 if max_val > 50.0 else 0.1 * max_val
    ys, xs = np.where(fg > threshold)
    if len(ys) > 0 and len(xs) > 0:
        fg = fg[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    h, w = fg.shape
    if h == 0 or w == 0:
        return Image.new("L", (28, 28), color=0)

    # Slight dilation/smoothing if extracted from a high-resolution crop so strokes don't become too thin
    crop_pil = Image.fromarray(fg.astype(np.uint8), mode="L")
    if max(h, w) > 40:
        crop_pil = crop_pil.filter(ImageFilter.MaxFilter(size=3))

    # Fit inside a 20x20 box while preserving aspect ratio (official MNIST protocol)
    scale = 20.0 / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized_pil = crop_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    resized = np.asarray(resized_pil, dtype=np.float32)
    if resized.max() > 0:
        resized = (resized / resized.max()) * 255.0

    # Place inside 28x28 black canvas aligned by center of mass at (13.5, 13.5)
    canvas = np.zeros((28, 28), dtype=np.float32)
    total_mass = float(resized.sum())
    if total_mass > 0.0:
        cy, cx = np.indices(resized.shape)
        com_y = float((cy * resized).sum() / total_mass)
        com_x = float((cx * resized).sum() / total_mass)
        top = int(round(13.5 - com_y))
        left = int(round(13.5 - com_x))
    else:
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2

    top = max(0, min(28 - new_h, top))
    left = max(0, min(28 - new_w, left))
    canvas[top : top + new_h, left : left + new_w] = resized

    return Image.fromarray(canvas.astype(np.uint8), mode="L")


def naive_preprocess_pil_to_mnist28(image: Image.Image) -> Image.Image:
    """Naive resize + invert without background suppression or center-of-mass bounding-box normalization."""
    gray = image.convert("L")
    fitted = ImageOps.fit(gray, (28, 28), method=Image.Resampling.LANCZOS)
    arr = np.asarray(fitted, dtype=np.float32)
    if arr.mean() > 127.0:
        fitted = ImageOps.invert(fitted)
    return fitted


def mnist28_to_tensor(mnist28_image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    return transform(mnist28_image)


def preprocess_handwritten_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    mnist28 = preprocess_pil_to_mnist28(image)
    return mnist28_to_tensor(mnist28)


def segment_student_number_image(
    image_path: str,
    dark_threshold: float = 115.0,
    border_fraction: float = 0.05,
    pad: int = 8,
) -> list[dict]:
    """Segment a multi-digit handwritten number photo left-to-right into individual digit crops."""
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape

    mask = arr < dark_threshold
    margin_y = int(border_fraction * h)
    margin_x = int(border_fraction * w)
    if margin_y > 0:
        mask[:margin_y, :] = False
        mask[h - margin_y :, :] = False
    if margin_x > 0:
        mask[:, :margin_x] = False
        mask[:, w - margin_x :] = False

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    y_min_all, y_max_all = int(ys.min()), int(ys.max())
    band = mask[y_min_all : y_max_all + 1, :]
    in_digit = band.sum(axis=0) > 0
    padded = np.concatenate([[0], in_digit.astype(int), [0]])
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    valid_spans: list[tuple[int, int]] = [
        (int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= 5
    ]

    segments: list[dict] = []
    for i, (s, e) in enumerate(valid_spans):
        ys_d = np.where(mask[:, s:e])[0]
        if len(ys_d) == 0:
            continue
        y0 = max(0, int(ys_d.min()) - pad)
        y1 = min(h, int(ys_d.max()) + 1 + pad)

        left_limit = (valid_spans[i - 1][1] + s) // 2 if i > 0 else 0
        right_limit = (e + valid_spans[i + 1][0]) // 2 if i + 1 < len(valid_spans) else w

        x0 = max(left_limit, s - pad)
        x1 = min(right_limit, e + pad)
        crop_pil = image.crop((x0, y0, x1, y1))
        mnist28_pil = preprocess_pil_to_mnist28(crop_pil)
        naive28_pil = naive_preprocess_pil_to_mnist28(crop_pil)
        segments.append(
            {
                "bbox": (x0, y0, x1, y1),
                "crop_pil": crop_pil,
                "mnist28_pil": mnist28_pil,
                "naive28_pil": naive28_pil,
            }
        )
    return segments


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MLPClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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


def evaluate_student_number(
    checkpoint_path: str,
    student_number_image: str,
    expected_number: str,
    output_dir: str,
    digits_save_dir: str = "./data/handwritten_digits/digits_28x28",
) -> dict:
    device = get_device()
    model = load_model_from_checkpoint(checkpoint_path, device)
    segments = segment_student_number_image(student_number_image)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_digits_path = Path(digits_save_dir)
    save_digits_path.mkdir(parents=True, exist_ok=True)

    expected_digits = [int(ch) for ch in expected_number.strip()]
    per_digit_results = []
    aligned_correct = 0
    naive_correct = 0

    with torch.no_grad():
        for idx, seg in enumerate(segments):
            true_digit = expected_digits[idx] if idx < len(expected_digits) else None

            # Save 28x28 MNIST-preprocessed digit image
            label_tag = f"_{true_digit}" if true_digit is not None else ""
            digit_filename = f"digit_{idx + 1:02d}{label_tag}.png"
            digit_save_file = save_digits_path / digit_filename
            seg["mnist28_pil"].save(digit_save_file)

            # Predict with MNIST-aligned preprocessing
            tensor_aligned = mnist28_to_tensor(seg["mnist28_pil"]).unsqueeze(0).to(device)
            logits_aligned = model(tensor_aligned)
            probs_aligned = torch.softmax(logits_aligned, dim=1).squeeze(0).cpu().numpy()
            pred_aligned = int(probs_aligned.argmax())
            conf_aligned = float(probs_aligned[pred_aligned])

            # Predict with naive preprocessing (for distribution shift analysis)
            tensor_naive = mnist28_to_tensor(seg["naive28_pil"]).unsqueeze(0).to(device)
            logits_naive = model(tensor_naive)
            probs_naive = torch.softmax(logits_naive, dim=1).squeeze(0).cpu().numpy()
            pred_naive = int(probs_naive.argmax())
            conf_naive = float(probs_naive[pred_naive])

            is_correct = (pred_aligned == true_digit) if true_digit is not None else None
            is_naive_correct = (pred_naive == true_digit) if true_digit is not None else None
            if is_correct:
                aligned_correct += 1
            if is_naive_correct:
                naive_correct += 1

            per_digit_results.append(
                {
                    "index": idx + 1,
                    "bbox": list(seg["bbox"]),
                    "saved_28x28_path": str(digit_save_file.as_posix()),
                    "true_digit": true_digit,
                    "predicted_digit": pred_aligned,
                    "confidence": conf_aligned,
                    "correct": is_correct,
                    "naive_predicted_digit": pred_naive,
                    "naive_confidence": conf_naive,
                    "naive_correct": is_naive_correct,
                }
            )

    n = len(segments)
    predicted_string = "".join(str(r["predicted_digit"]) for r in per_digit_results)
    naive_predicted_string = "".join(str(r["naive_predicted_digit"]) for r in per_digit_results)
    accuracy = aligned_correct / n if n > 0 else 0.0
    naive_accuracy = naive_correct / n if n > 0 else 0.0

    # Save visual summary plot
    if n > 0:
        fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.8))
        if n == 1:
            axes = np.expand_dims(axes, axis=1)
        for idx, (seg, res) in enumerate(zip(segments, per_digit_results)):
            ax_raw = axes[0, idx]
            ax_raw.imshow(seg["crop_pil"], cmap="gray", vmin=0, vmax=255)
            ax_raw.set_title(f"Gerçek: {res['true_digit']}", fontsize=9)
            ax_raw.axis("off")

            ax_mnist = axes[1, idx]
            ax_mnist.imshow(seg["mnist28_pil"], cmap="gray", vmin=0, vmax=255)
            color = "green" if res["correct"] else "red"
            ax_mnist.set_title(
                f"Tahmin: {res['predicted_digit']}\n(%{res['confidence'] * 100:.1f})",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
            ax_mnist.axis("off")

        fig.suptitle(
            f"Öğrenci Numarası ({expected_number}) El Yazısı Tahminleri — Doğruluk: {aligned_correct}/{n} (%{accuracy * 100:.1f})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_path / "student_number_predictions.png", dpi=150)
        plt.close(fig)

        # Also save a figure comparing Naive vs MNIST-Aligned preprocessing (Distribution Shift illustration)
        fig2, axes2 = plt.subplots(3, n, figsize=(1.5 * n, 5.0))
        if n == 1:
            axes2 = np.expand_dims(axes2, axis=1)
        for idx, (seg, res) in enumerate(zip(segments, per_digit_results)):
            axes2[0, idx].imshow(seg["crop_pil"], cmap="gray", vmin=0, vmax=255)
            axes2[0, idx].set_title(f"Ham Kesit\n({res['true_digit']})", fontsize=8)
            axes2[0, idx].axis("off")

            c_naive = "green" if res["naive_correct"] else "crimson"
            axes2[1, idx].imshow(seg["naive28_pil"], cmap="gray", vmin=0, vmax=255)
            axes2[1, idx].set_title(
                f"Basit Tersleme\nTahmin: {res['naive_predicted_digit']} (%{res['naive_confidence']*100:.0f})",
                fontsize=8,
                color=c_naive,
            )
            axes2[1, idx].axis("off")

            c_aligned = "green" if res["correct"] else "crimson"
            axes2[2, idx].imshow(seg["mnist28_pil"], cmap="gray", vmin=0, vmax=255)
            axes2[2, idx].set_title(
                f"MNIST Uyumlu\nTahmin: {res['predicted_digit']} (%{res['confidence']*100:.0f})",
                fontsize=8,
                color=c_aligned,
            )
            axes2[2, idx].axis("off")

        fig2.suptitle(
            f"Distribution Shift Analizi: Basit Ön İşleme (%{naive_accuracy*100:.1f}) vs. MNIST Uyumlu Ön İşleme (%{accuracy*100:.1f})",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_path / "preprocessing_comparison.png", dpi=150)
        plt.close(fig2)

    summary = {
        "student_number_image": student_number_image,
        "expected_number": expected_number,
        "predicted_number": predicted_string,
        "total_digits": n,
        "correct_digits": aligned_correct,
        "accuracy": accuracy,
        "naive_predicted_number": naive_predicted_string,
        "naive_correct_digits": naive_correct,
        "naive_accuracy": naive_accuracy,
        "digits": per_digit_results,
    }
    (out_path / "student_number_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict digits for handwritten images using a trained MLP checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student-number-image", default="", help="Path to multi-digit student number image.")
    parser.add_argument("--expected-number", default="24120205049", help="Expected student number string.")
    parser.add_argument("--output-dir", default="./outputs/handwriting", help="Directory for evaluation outputs.")
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()

    if args.student_number_image:
        summary = evaluate_student_number(
            checkpoint_path=args.checkpoint,
            student_number_image=args.student_number_image,
            expected_number=args.expected_number,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "expected_number": summary["expected_number"],
                    "predicted_number": summary["predicted_number"],
                    "correct_digits": f"{summary['correct_digits']}/{summary['total_digits']}",
                    "accuracy": summary["accuracy"],
                    "naive_predicted_number": summary["naive_predicted_number"],
                    "naive_accuracy": summary["naive_accuracy"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.images:
        predictions = predict_images(args.checkpoint, args.images)
        print(json.dumps(predictions, indent=2, ensure_ascii=False))
    else:
        parser.error("Provide either --student-number-image or one or more single-digit image paths.")


if __name__ == "__main__":
    main()
