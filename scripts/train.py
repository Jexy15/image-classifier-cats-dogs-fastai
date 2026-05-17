from __future__ import annotations

import os
from pathlib import Path

from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    URLs,
    error_rate,
    get_image_files,
    resnet34,
    untar_data,
    vision_learner,
)


def is_cat(filename: str) -> bool:
    """Return True if filename indicates a cat breed (uppercase first letter in Pets)."""
    return Path(filename).name[0].isupper()


def train(output_path: Path, epochs: int = 1) -> None:
    dataset_override = os.environ.get("PETS_DATASET_PATH")
    if dataset_override:
        path = Path(dataset_override)
    else:
        path = untar_data(URLs.PETS) / "images"
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        seed=42,
        label_func=is_cat,
        item_tfms=Resize(224),
        num_workers=num_workers,
    )
    learner = vision_learner(dls, resnet34, metrics=error_rate)
    learner.fine_tune(epochs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    learner.export(output_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    # Binary cat/dog classifier derived from Oxford-IIIT Pets breed naming.
    model_path = root / "model" / "cat_dog_classifier.pkl"
    train(model_path)
