"""
Fine-tuning pipeline for BERT on fake news classification.
Includes mixed-precision (AMP), gradient clipping, dynamic DataLoader batching,
learning rate scheduling, early stopping, and artifact checkpointing.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data_loader import load_combined_dataset
from src.model import (
    DEFAULT_MODEL_NAME,
    get_device,
    get_model_and_tokenizer,
    save_model_and_tokenizer,
)
from src.preprocessing import (
    NewsDataset,
    create_stratified_splits,
    prepare_dataframe,
)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    max_grad_norm: float = 1.0,
) -> float:
    """Runs a single training epoch with optional mixed-precision."""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision context if CUDA is active
        use_amp = (device.type == "cuda") and (scaler is not None)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate_epoch(
    model,
    dataloader: DataLoader,
    device,
) -> Tuple[float, float, float]:
    """
    Evaluates model on validation data.
    Returns: (val_loss, accuracy, macro_f1)
    """
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def train_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 2,
    seed: int = 42,
    sample_size: Optional[int] = None,
    output_dir: str = "./artifacts/model",
    metrics_dir: str = "./artifacts/metrics",
    strip_dateline: bool = True,
) -> Dict:
    """
    End-to-end training pipeline.
    """
    set_seed(seed)
    device, device_desc = get_device()
    print(f"Using compute device: {device_desc}")

    out_path = Path(output_dir).resolve()
    met_path = Path(metrics_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    met_path.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing dataset...")
    df = load_combined_dataset(shuffle=True, random_state=seed)
    df_clean = prepare_dataframe(df, strip_dateline=strip_dateline, drop_duplicates=True)

    if sample_size and sample_size < len(df_clean):
        print(f"Sampling {sample_size} articles for fast execution/verification...")
        df_clean = df_clean.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    train_df, val_df, test_df = create_stratified_splits(
        df_clean, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=seed
    )
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Save test set for clean downstream evaluation
    test_csv_path = met_path / "test_set.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved held-out test split to {test_csv_path}")

    # Initialize model and tokenizer
    print(f"Loading pretrained {model_name}...")
    model, tokenizer = get_model_and_tokenizer(model_name=model_name)
    model.to(device)

    # Datasets and DataLoaders
    train_dataset = NewsDataset(
        texts=train_df["formatted_text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = NewsDataset(
        texts=val_df["formatted_text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer and Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler = torch.amp.GradScaler('cuda') if (device.type == "cuda") else None

    # Training Loop with Early Stopping
    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    start_time = time.time()

    print(f"\n--- Starting Training ({epochs} epochs, {len(train_loader)} batches/epoch) ---")

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
        )
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
        )
        ep_duration = time.time() - ep_start

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "duration_seconds": round(ep_duration, 1),
        }
        history.append(log_entry)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {ep_duration:.1f}s"
        )

        # Checkpoint if best validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"  * New best validation F1 ({val_f1:.4f})! Saving model checkpoint...")
            save_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                output_dir=out_path,
                extra_config={
                    "max_length": max_length,
                    "best_val_f1": best_val_f1,
                    "best_epoch": epoch,
                    "strip_dateline": strip_dateline,
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.1f}s. Best Val F1: {best_val_f1:.4f}")

    # Save training history
    training_summary = {
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs_completed": len(history),
        "total_duration_seconds": round(total_duration, 1),
        "device_used": device_desc,
        "best_val_f1": round(best_val_f1, 4),
        "dataset_split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "history": history,
    }
    with open(met_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    return training_summary


if __name__ == "__main__":
    train_model()
