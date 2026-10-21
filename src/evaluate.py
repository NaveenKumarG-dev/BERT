"""
Comprehensive evaluation module for trained BERT model.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
and full ROC/PR curve points, saving structured JSON artifacts for Streamlit.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch.utils.data import DataLoader

from src.model import get_device, load_model_and_tokenizer
from src.preprocessing import NewsDataset


def evaluate_model(
    model_dir: Union[str, Path] = "./artifacts/model",
    test_data_path: Optional[Union[str, Path]] = None,
    metrics_dir: Union[str, Path] = "./artifacts/metrics",
    batch_size: int = 16,
    max_length: int = 256,
) -> Dict:
    """
    Evaluates saved BERT checkpoint on held-out test set and saves all metrics.
    """
    model_path = Path(model_dir).resolve()
    metrics_path = Path(metrics_dir).resolve()
    metrics_path.mkdir(parents=True, exist_ok=True)

    if test_data_path is None:
        test_data_path = metrics_path / "test_set.csv"

    test_file = Path(test_data_path).resolve()
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test set not found at {test_file}. Please run training first to generate test_set.csv."
        )

    print(f"Loading held-out test dataset from {test_file}...")
    test_df = pd.read_csv(test_file)
    print(f"Test set size: {len(test_df)} samples")

    device, device_desc = get_device()
    print(f"Loading trained model from {model_path} onto {device_desc}...")
    model, tokenizer, metadata = load_model_and_tokenizer(model_path, device=device)

    # Use max_length from metadata if present
    seq_len = metadata.get("max_length", max_length)

    test_dataset = NewsDataset(
        texts=test_df["formatted_text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=seq_len,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    all_preds = []
    all_probs = []
    all_labels = []

    print("Running inference across test batches...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs[:, 1].tolist())  # Probability of Fake (class 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute core performance metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_binary = float(precision_score(y_true, y_pred, pos_label=1))
    rec_binary = float(recall_score(y_true, y_pred, pos_label=1))
    f1_binary = float(f1_score(y_true, y_pred, pos_label=1))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred).tolist()
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=["Real (0)", "Fake (1)"],
        output_dict=True,
    )

    # Compute ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc,
    }

    # Compute Precision-Recall Curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_data = {
        "precision": precisions.tolist(),
        "recall": recalls.tolist(),
    }

    metrics_summary = {
        "accuracy": round(acc, 4),
        "precision_fake": round(prec_binary, 4),
        "recall_fake": round(rec_binary, 4),
        "f1_fake": round(f1_binary, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "test_sample_count": len(test_df),
        "class_distribution": {
            "Real": int((y_true == 0).sum()),
            "Fake": int((y_true == 1).sum()),
        },
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    # Save metrics JSON files
    with open(metrics_path / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(metrics_path / "roc_curve.json", "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)

    with open(metrics_path / "pr_curve.json", "w", encoding="utf-8") as f:
        json.dump(pr_data, f, indent=2)

    print("\n--- Test Set Evaluation Summary ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec_binary:.4f}")
    print(f"Recall:    {rec_binary:.4f}")
    print(f"F1 Score:  {f1_binary:.4f} (Macro F1: {f1_macro:.4f})")
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Metrics saved to {metrics_path}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_model()
