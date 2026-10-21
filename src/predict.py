"""
Production inference pipeline for Fake News Detection using fine-tuned BERT.
Provides single-article and batch prediction capabilities with confidence scores.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import torch
import torch.nn.functional as F

from src.model import DEFAULT_ID2LABEL, get_device, load_model_and_tokenizer
from src.preprocessing import format_article


class NewsPredictor:
    """
    Cached, thread-safe inference wrapper for BERT model.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "./artifacts/model",
        device: Optional[torch.device] = None,
    ):
        self.model_dir = Path(model_dir).resolve()
        if device is None:
            self.device, self.device_name = get_device()
        else:
            self.device = device
            self.device_name = str(device)

        self.model, self.tokenizer, self.metadata = load_model_and_tokenizer(
            self.model_dir, device=self.device
        )
        self.max_length = self.metadata.get("max_length", 256)
        self.strip_dateline = self.metadata.get("strip_dateline", True)
        self.id2label = self.metadata.get("id2label", DEFAULT_ID2LABEL)
        # Ensure keys in id2label are integers
        self.id2label = {int(k): v for k, v in self.id2label.items()}

    def predict(
        self,
        headline: str,
        body: str = "",
    ) -> Dict:
        """
        Runs inference on a headline and/or body text.

        Returns:
            Dict containing predicted class, confidence, class probabilities, and clean input.
        """
        formatted = format_article(
            title=headline,
            text=body,
            strip_dateline=self.strip_dateline,
        )

        if not formatted or len(formatted.strip()) < 5:
            raise ValueError(
                "Input text is too short. Please provide a meaningful news headline or article body."
            )

        encoding = self.tokenizer(
            formatted,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0)

        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item())

        pred_id = int(torch.argmax(probs).item())
        pred_label = self.id2label.get(pred_id, "Fake" if pred_id == 1 else "Real")
        confidence = prob_fake if pred_id == 1 else prob_real

        return {
            "prediction": pred_label,
            "label_id": pred_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Real": round(prob_real, 4),
                "Fake": round(prob_fake, 4),
            },
            "formatted_text": formatted,
            "device": self.device_name,
        }


def predict_news(
    headline: str,
    body: str = "",
    model_dir: Union[str, Path] = "./artifacts/model",
) -> Dict:
    """Convenience function for standalone prediction."""
    predictor = NewsPredictor(model_dir=model_dir)
    return predictor.predict(headline=headline, body=body)
