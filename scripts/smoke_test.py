#!/usr/bin/env python3
"""Minimal CPU smoke test for the archived region-similarity code."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ARViT.ARViT import ARViT  # noqa: E402
from losses.attention_loss import ARViT_Loss  # noqa: E402


def load_batch(use_imagenette: bool, batch_size: int = 2) -> torch.Tensor:
    if not use_imagenette:
        return torch.rand(batch_size, 3, 256, 256)

    from fastai.data.external import URLs, untar_data

    root = Path(untar_data(URLs.IMAGENETTE_160))
    files = sorted((root / "train").glob("*/*"))
    files = [path for path in files if path.suffix.lower() in {".jpeg", ".jpg", ".png"}]
    if len(files) < batch_size:
        raise RuntimeError(f"Imagenette acquisition returned only {len(files)} images")

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    return torch.stack([transform(Image.open(path).convert("RGB")) for path in files[:batch_size]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenette", action="store_true", help="download Imagenette-160 and use one batch")
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.set_num_threads(2)
    batch_size = 2

    model = ARViT(
        num_encoder_layers=1,
        nhead=4,
        num_classes=3,
        batch_size=batch_size,
        hidden_dim=32,
        image_h=256,
        image_w=256,
        grid_l=16,
        gm_patch=16,
    )
    model.train()

    images = load_batch(args.imagenette, batch_size=batch_size)
    labels = torch.tensor([0, 1], dtype=torch.long)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    optimizer.zero_grad()

    outputs = model(images)
    logits, reduced_attention, raw_attention, gram_distance = outputs

    assert logits.shape == (batch_size, 3)
    assert len(reduced_attention) == 1
    assert len(raw_attention) == 1
    assert reduced_attention[0].shape == (batch_size, 256, 256)
    assert raw_attention[0].shape == (batch_size, 256, 256)
    assert gram_distance.shape == (batch_size, 256, 256)
    assert torch.isfinite(gram_distance).all()
    assert float(gram_distance.min()) >= -1e-6
    assert float(gram_distance.max()) <= 1.0 + 1e-6

    loss = ARViT_Loss(layer=0, bias=-0.17, lambda_=0.001)(outputs, labels)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite historical loss: {loss}")
    loss.backward()
    optimizer.step()

    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients:
        raise RuntimeError("No trainable parameter received a gradient")

    source = "Imagenette-160" if args.imagenette else "synthetic input"
    print(f"PASS: region extraction, Gram distances, attention loss, and one optimizer step using {source}")
    print(f"torch={torch.__version__}; loss={float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
