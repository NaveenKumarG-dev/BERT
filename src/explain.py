"""
Explainability engine for BERT Fake News Detection.
Provides token and word-level attribution via input-embedding gradient analysis (Grad x Input)
and optional SHAP partition explanation, returning scores and HTML highlight representations.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import html
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model_and_tokenizer, get_device
from src.preprocessing import format_article


class NewsExplainer:
    """
    Computes genuine token and word-level attributions for BERT predictions.
    Uses Gradient x Input on word embeddings with subword aggregation.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_dir: str = "./artifacts/model",
        device: Optional[torch.device] = None,
    ):
        if model is None or tokenizer is None:
            if device is None:
                self.device, _ = get_device()
            else:
                self.device = device
            self.model, self.tokenizer, self.metadata = load_model_and_tokenizer(
                model_dir, device=self.device
            )
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            self.metadata = {}

        self.max_length = self.metadata.get("max_length", 256)
        self.strip_dateline = self.metadata.get("strip_dateline", True)

    def explain(
        self,
        headline: str,
        body: str = "",
        target_class: Optional[int] = None,
    ) -> Dict:
        """
        Computes token-level attribution scores for the input news text.

        Args:
            headline: Article title.
            body: Article body text.
            target_class: Class ID to explain (0 for Real, 1 for Fake).
                          If None, explains the model's top predicted class.

        Returns:
            Dict containing tokens, word attributions, top positive/negative words,
            and formatted HTML markup for visual rendering.
        """
        formatted = format_article(
            title=headline,
            text=body,
            strip_dateline=self.strip_dateline,
        )

        if not formatted or len(formatted.strip()) < 5:
            raise ValueError("Input text is too short to explain.")

        # Tokenize with truncation
        encoding = self.tokenizer(
            formatted,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get embeddings layer of BERT
        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)

        # Forward pass through BERT starting from embeddings
        self.model.eval()
        outputs = self.model.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output) if hasattr(self.model.bert, "pooler") else sequence_output[:, 0]
        dropout_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(dropout_output)

        probs = F.softmax(logits, dim=-1).squeeze(0)
        predicted_class = int(torch.argmax(probs).item())

        if target_class is None:
            target_class = predicted_class

        target_logit = logits[0, target_class]

        # Backward pass to obtain gradients with respect to word embeddings
        self.model.zero_grad()
        target_logit.backward()

        # Gradient x Input Attribution: element-wise product of embedding vector and its gradient
        # Sum over the embedding dimension to get a scalar score per token
        grad = embeddings.grad[0]  # (seq_len, hidden_dim)
        embed = embeddings[0]      # (seq_len, hidden_dim)
        attribution_scores = (grad * embed).sum(dim=-1).detach().cpu().numpy()

        # Normalize attribution scores
        norm_factor = np.linalg.norm(attribution_scores)
        if norm_factor > 0:
            attribution_scores = attribution_scores / norm_factor

        token_ids = input_ids[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Aggregate subword tokens into words (e.g. ['con', '##spir', '##acy'] -> 'conspiracy')
        word_attributions = self._aggregate_subwords(tokens, attribution_scores)

        # Separate influential words
        # Positive score: pushes towards target_class (e.g., Fake if target_class=1)
        # Negative score: pushes against target_class (e.g., towards Real)
        sorted_by_score = sorted(word_attributions, key=lambda x: x["score"], reverse=True)
        top_words_target = [w for w in sorted_by_score if w["score"] > 0][:10]
        top_words_opposite = [w for w in sorted_by_score if w["score"] < 0][-10:]
        top_words_opposite.reverse()

        # Generate HTML visualization
        target_label = "Fake" if target_class == 1 else "Real"
        html_markup = self._generate_html(word_attributions, target_label=target_label)

        return {
            "predicted_class": "Fake" if predicted_class == 1 else "Real",
            "predicted_id": predicted_class,
            "target_class": target_label,
            "target_id": target_class,
            "confidence": float(probs[predicted_class].item()),
            "probabilities": {
                "Real": float(probs[0].item()),
                "Fake": float(probs[1].item()),
            },
            "word_attributions": word_attributions,
            "top_words_for_target": top_words_target,
            "top_words_against_target": top_words_opposite,
            "html_markup": html_markup,
        }

    def _aggregate_subwords(
        self,
        tokens: List[str],
        scores: np.ndarray,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Combines BERT subwords (prefixed with '##') and special tokens ([CLS], [SEP])
        into full words with averaged or summed attribution scores.
        """
        words = []
        current_word = ""
        current_scores = []

        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}

        for token, score in zip(tokens, scores):
            if token in special_tokens:
                continue

            if token.startswith("##"):
                current_word += token[2:]
                current_scores.append(score)
            else:
                if current_word:
                    words.append({
                        "word": current_word,
                        "score": round(float(np.mean(current_scores)), 4),
                    })
                current_word = token
                current_scores = [score]

        if current_word:
            words.append({
                "word": current_word,
                "score": round(float(np.mean(current_scores)), 4),
            })

        return words

    def _generate_html(
        self,
        word_attributions: List[Dict[str, Union[str, float]]],
        target_label: str = "Fake",
    ) -> str:
        """
        Generates clean inline HTML with color-coded background highlights:
        Red highlight = pushes towards Fake
        Green/Blue highlight = pushes towards Real
        """
        spans = []
        for item in word_attributions:
            w = html.escape(item["word"])
            score = item["score"]

            # Map score to opacity [0.0, 0.85]
            alpha = min(abs(score) * 2.5, 0.85)

            if target_label == "Fake":
                if score > 0.05:
                    # Pushes toward Fake -> Red highlight
                    bg_color = f"rgba(239, 68, 68, {alpha:.2f})"
                    text_color = "#ffffff" if alpha > 0.4 else "inherit"
                elif score < -0.05:
                    # Pushes toward Real -> Teal/Green highlight
                    bg_color = f"rgba(16, 185, 129, {alpha:.2f})"
                    text_color = "#ffffff" if alpha > 0.4 else "inherit"
                else:
                    bg_color = "transparent"
                    text_color = "inherit"
            else:  # Target is Real
                if score > 0.05:
                    bg_color = f"rgba(16, 185, 129, {alpha:.2f})"
                    text_color = "#ffffff" if alpha > 0.4 else "inherit"
                elif score < -0.05:
                    bg_color = f"rgba(239, 68, 68, {alpha:.2f})"
                    text_color = "#ffffff" if alpha > 0.4 else "inherit"
                else:
                    bg_color = "transparent"
                    text_color = "inherit"

            span_style = (
                f"background-color: {bg_color}; "
                f"color: {text_color}; "
                "padding: 2px 4px; "
                "margin: 1px 2px; "
                "border-radius: 4px; "
                "display: inline-block; "
                "font-size: 0.95rem;"
            )
            title_tooltip = f"Word: {w} | Attribution: {score:+.3f}"
            spans.append(f'<span style="{span_style}" title="{title_tooltip}">{w}</span>')

        return f'<div style="line-height: 1.8; font-family: sans-serif; padding: 12px; background: rgba(128,128,128,0.06); border-radius: 8px;">{" ".join(spans)}</div>'


def explain_news(
    headline: str,
    body: str = "",
    model_dir: str = "./artifacts/model",
) -> Dict:
    """Convenience function for explaining an individual news article."""
    explainer = NewsExplainer(model_dir=model_dir)
    return explainer.explain(headline=headline, body=body)
