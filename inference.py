from __future__ import annotations

import json
import os
from pathlib import Path

from fastai.vision.all import PILImage, load_learner

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "cat_dog_classifier.pkl"
LABELS_PATH = BASE_DIR / "model" / "labels.json"


def _load_labels() -> dict[str, str]:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


LABELS = _load_labels()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run scripts/train.py to export it."
        )
    if os.name != "nt":
        # Work around WindowsPath objects inside pickled fastai exports.
        try:
            import pathlib

            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined]
        except Exception:
            pass
    return load_learner(MODEL_PATH, cpu=True)


def predict_image(learner, image) -> tuple[str, float]:
    pil_image = PILImage.create(image)
    pred_class, pred_idx, probs = learner.predict(pil_image)
    raw_label = str(pred_class)
    label = LABELS.get(raw_label, raw_label)
    confidence = float(probs[int(pred_idx)])
    return label, confidence
