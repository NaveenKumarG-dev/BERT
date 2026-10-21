"""
Unit tests for text preprocessing, dateline leakage cleaning, and dataset splitting.
"""

import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    create_stratified_splits,
    format_article,
    prepare_dataframe,
    remove_dateline,
)


def test_clean_text_normalizes_whitespace():
    raw = "  Breaking   news:   Something    happened! \n\n  "
    cleaned = clean_text(raw, strip_dateline=False)
    assert cleaned == "Breaking news: Something happened!"


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text(float("nan")) == ""


def test_remove_dateline_reuters_header():
    sample = "WASHINGTON (Reuters) - The president signed a new executive order today."
    cleaned = remove_dateline(sample)
    assert cleaned == "The president signed a new executive order today."


def test_remove_dateline_leaves_normal_text_intact():
    sample = "A new study reveals significant shifts in public opinion."
    cleaned = remove_dateline(sample)
    assert cleaned == sample


def test_format_article_concatenation():
    title = "Historic Summit Concludes"
    body = "World leaders reached a mutual agreement on economic targets."
    formatted = format_article(title, body)
    assert formatted == "Historic Summit Concludes. World leaders reached a mutual agreement on economic targets."


def test_prepare_dataframe_drops_duplicates_and_creates_column():
    data = {
        "title": [
            "News A",
            "News A",  # Duplicate
            "News B",
            "",
        ],
        "text": [
            "WASHINGTON (Reuters) - Content A here.",
            "WASHINGTON (Reuters) - Content A here.",  # Duplicate
            "Content B is here and has sufficient length.",
            "",  # Short/empty
        ],
        "label": [0, 0, 1, 1],
    }
    df = pd.DataFrame(data)
    processed = prepare_dataframe(df, strip_dateline=True, drop_duplicates=True)

    # 4 rows -> 2 unique valid rows
    assert len(processed) == 2
    assert "formatted_text" in processed.columns
    assert "Content A here." in processed.loc[0, "formatted_text"]


def test_create_stratified_splits():
    data = {
        "title": [f"Title {i}" for i in range(100)],
        "text": [f"Body text content for article {i} with sufficient words." for i in range(100)],
        "label": [0] * 50 + [1] * 50,
    }
    df = prepare_dataframe(pd.DataFrame(data))
    train, val, test = create_stratified_splits(
        df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
    )

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Ensure zero overlap
    train_texts = set(train["formatted_text"])
    val_texts = set(val["formatted_text"])
    test_texts = set(test["formatted_text"])

    assert len(train_texts.intersection(val_texts)) == 0
    assert len(train_texts.intersection(test_texts)) == 0
    assert len(val_texts.intersection(test_texts)) == 0
