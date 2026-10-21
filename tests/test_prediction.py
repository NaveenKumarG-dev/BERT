"""
Tests for prediction pipeline, probability normalization, and input validation.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from src.predict import NewsPredictor, predict_news


def test_predict_empty_input_raises_value_error():
    with pytest.raises(ValueError, match="Input text is too short"):
        # We test input validation even before model inference
        predictor = object.__new__(NewsPredictor)
        predictor.strip_dateline = True
        predictor.predict("", "")


def test_prediction_output_structure_and_probabilities_sum_to_one():
    # Test with mocked model and tokenizer to verify output logic and probability calibration
    predictor = object.__new__(NewsPredictor)
    predictor.device = torch.device("cpu")
    predictor.device_name = "CPU"
    predictor.max_length = 128
    predictor.strip_dateline = True
    predictor.id2label = {0: "Real", 1: "Fake"}

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    predictor.tokenizer = mock_tokenizer

    # Mock model returning logits
    mock_model = MagicMock()
    # Logits where class 1 (Fake) is higher
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[-1.5, 2.5]])
    mock_model.return_value = mock_outputs
    predictor.model = mock_model

    result = predictor.predict(
        headline="Shocking Revelation Regarding UFOs",
        body="Anonymous sources claim secret bases exist on the moon.",
    )

    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "Real" in result["probabilities"]
    assert "Fake" in result["probabilities"]

    assert result["prediction"] == "Fake"
    assert result["label_id"] == 1

    prob_sum = result["probabilities"]["Real"] + result["probabilities"]["Fake"]
    assert abs(prob_sum - 1.0) < 1e-3
    assert result["confidence"] == result["probabilities"]["Fake"]
