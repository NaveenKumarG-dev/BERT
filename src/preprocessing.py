"""
Text preprocessing and dataset preparation pipeline for BERT model.
Handles text cleaning, publisher leakage removal, formatting,
stratified train/val/test splitting, and PyTorch Dataset creation.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Allows module to be imported before torch is fully installed
    torch = None
    Dataset = object


# Regex pattern to match news agency datelines like "WASHINGTON (Reuters) - " or "LONDON (Reuters) -"
REUTERS_DATELINE_PATTERN = re.compile(
    r"^[A-Za-z\s,/\.\-]+(?:\([A-Za-z\s,]+\))?\s*[-–—]\s*",
    re.MULTILINE
)


def remove_dateline(text: str) -> str:
    """
    Removes news agency datelines (e.g., 'WASHINGTON (Reuters) - ')
    which are present in True.csv and cause trivial data leakage shortcuts.
    """
    if not isinstance(text, str):
        return ""
    # Strip leading whitespace
    text = text.strip()
    # If the text starts with a dateline like "WASHINGTON (Reuters) -", strip it
    cleaned = REUTERS_DATELINE_PATTERN.sub("", text, count=1)
    return cleaned.strip()


def clean_text(
    text: Union[str, float],
    strip_dateline: bool = True,
) -> str:
    """
    Cleans an individual text entry safely:
    - Fills missing/null with empty string
    - Normalizes whitespace
    - Optionally removes publisher dateline to prevent data leakage shortcuts
    - Preserves punctuation and casing needed by BERT
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.strip()
    if strip_dateline:
        text = remove_dateline(text)

    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_article(
    title: Union[str, float],
    text: Union[str, float],
    strip_dateline: bool = True,
) -> str:
    """
    Combines article title and body text into a single unified string
    for BERT input, formatted consistently as '{clean_title}. {clean_text}'.
    """
    clean_t = clean_text(title, strip_dateline=False)
    clean_b = clean_text(text, strip_dateline=strip_dateline)

    if clean_t and clean_b:
        # If title doesn't end with punctuation, add period
        sep = "" if clean_t[-1] in ".!?:;" else "."
        return f"{clean_t}{sep} {clean_b}"
    elif clean_t:
        return clean_t
    elif clean_b:
        return clean_b
    return ""


def prepare_dataframe(
    df: pd.DataFrame,
    strip_dateline: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Prepares raw news DataFrame:
    1. Removes duplicate rows based on title and text.
    2. Combines title and text into a 'formatted_text' column.
    3. Filters out rows where formatted_text is empty or too short.
    """
    processed = df.copy()

    # Fill NaNs
    processed["title"] = processed["title"].fillna("")
    processed["text"] = processed["text"].fillna("")

    if drop_duplicates:
        # Drop duplicates on combined title + text to prevent cross-split leakage
        processed = processed.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    # Create unified formatted_text
    processed["formatted_text"] = [
        format_article(t, b, strip_dateline=strip_dateline)
        for t, b in zip(processed["title"], processed["text"])
    ]

    # Filter out empty or trivially short entries (< 10 chars)
    processed = processed[processed["formatted_text"].str.len() >= 10].reset_index(drop=True)

    return processed


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    stratify_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train, validation, and test sets using stratified sampling.
    Guarantees no overlap in index.
    """
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-5, f"Ratios must sum to 1.0, got {total}"

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col in df.columns else None,
    )

    # Second split: val vs test
    test_share_of_temp = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share_of_temp,
        random_state=random_state,
        stratify=temp_df[stratify_col] if stratify_col in temp_df.columns else None,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class NewsDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning BERT on news classification.
    Returns tokenized tensors: input_ids, attention_mask, and label.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 256,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union["torch.Tensor", int]]:
        text = str(self.texts[idx])

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        else:
            item = {"text": text}

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long) if torch else self.labels[idx]

        return item
