"""
BERT model initialization, device management, and serialization.
Wraps Hugging Face AutoModelForSequenceClassification and AutoTokenizer.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_ID2LABEL = {0: "Real", 1: "Fake"}
DEFAULT_LABEL2ID = {"Real": 0, "Fake": 1}


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA GPU if available, else CPU).
    Safely handles environments before or after torch is imported.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return torch.device("cuda"), f"GPU: {device_name}"
        return torch.device("cpu"), "CPU"
    except Exception:
        return "cpu", "CPU"


def get_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 2,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
):
    """
    Initializes pretrained BERT tokenizer and sequence classification model.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if id2label is None:
        id2label = DEFAULT_ID2LABEL
    if label2id is None:
        label2id = DEFAULT_LABEL2ID

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer


def save_model_and_tokenizer(
    model,
    tokenizer,
    output_dir: Union[str, Path],
    extra_config: Optional[dict] = None,
) -> Path:
    """
    Persists model weights, tokenizer vocab/config, and metadata to disk.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Save Hugging Face model and tokenizer artifacts
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Save additional metadata
    meta = {
        "model_type": "bert-base-uncased",
        "id2label": DEFAULT_ID2LABEL,
        "label2id": DEFAULT_LABEL2ID,
    }
    if extra_config:
        meta.update(extra_config)

    with open(out_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_path


def load_model_and_tokenizer(
    model_dir: Union[str, Path],
    device=None,
):
    """
    Loads saved model, tokenizer, and metadata from directory.
    Places model in eval mode on the requested device.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model_path = Path(model_dir).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    if device is None:
        device, _ = get_device()

    model.to(device)
    model.eval()

    meta_file = model_path / "model_metadata.json"
    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, tokenizer, metadata
