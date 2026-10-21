"""
Streamlit Page 1: Comprehensive Model Performance & Evaluation Dashboard.
Displays genuine evaluation metrics, confusion matrix, ROC/PR curves, and training history.
"""

import json
from pathlib import Path
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Model Performance | BERT",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation & Benchmark Metrics")
st.markdown(
    "Rigorous evaluation of the fine-tuned BERT classifier on the held-out test split. "
    "All statistics and curves shown below are generated from genuine evaluation runs."
)

metrics_dir = Path("./artifacts/metrics").resolve()
metrics_file = metrics_dir / "evaluation_metrics.json"
history_file = metrics_dir / "training_history.json"
roc_file = metrics_dir / "roc_curve.json"
pr_file = metrics_dir / "pr_curve.json"

if not metrics_file.exists():
    st.warning("⚠️ No evaluation metrics found in `artifacts/metrics/evaluation_metrics.json`.")
    st.info(
        "To generate real evaluation metrics, run the evaluation pipeline on your trained model:\n"
        "```bash\n"
        "python src/evaluate.py\n"
        "```\n"
        "If you have not trained the model yet, run:\n"
        "```bash\n"
        "python src/train.py\n"
        "```"
    )
else:
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Headline KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with col2:
        st.metric("Macro F1-Score", f"{metrics['f1_macro'] * 100:.2f}%")
    with col3:
        st.metric("Fake News Precision", f"{metrics['precision_fake'] * 100:.2f}%")
    with col4:
        st.metric("Fake News Recall", f"{metrics['recall_fake'] * 100:.2f}%")
    with col5:
        roc_val = metrics.get("roc_auc")
        st.metric("ROC-AUC", f"{roc_val:.4f}" if roc_val is not None else "N/A")

    st.divider()

    # 2. Confusion Matrix & Classification Report
    col_cm, col_cr = st.columns([1, 1.2])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = metrics.get("confusion_matrix")
        if cm and len(cm) == 2:
            import plotly.graph_objects as go

            z = cm
            x = ["Pred: Real", "Pred: Fake"]
            y = ["True: Real", "True: Fake"]
            z_text = [[str(val) for val in row] for row in z]

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="Blues",
                    showscale=True,
                    text=z_text,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                )
            )
            fig_cm.update_layout(
                margin=dict(t=30, l=10, r=10, b=10),
                height=320,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp = cm[0]
            fn, tp = cm[1]
            st.caption(
                f"True Negatives: {tn} | False Positives: {fp} | "
                f"False Negatives: {fn} | True Positives: {tp}"
            )

    with col_cr:
        st.subheader("Detailed Classification Report")
        clf_rep = metrics.get("classification_report", {})
        if clf_rep:
            import pandas as pd
            # Format as clean DataFrame
            rows = []
            for k, v in clf_rep.items():
                if isinstance(v, dict):
                    rows.append({
                        "Class": k,
                        "Precision": f"{v.get('precision', 0):.4f}",
                        "Recall": f"{v.get('recall', 0):.4f}",
                        "F1-Score": f"{v.get('f1-score', 0):.4f}",
                        "Support": int(v.get("support", 0)),
                    })
            df_rep = pd.DataFrame(rows)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

    st.divider()

    # 3. ROC & Precision-Recall Curves
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.subheader("Receiver Operating Characteristic (ROC)")
        if roc_file.exists():
            with open(roc_file, "r", encoding="utf-8") as f:
                roc_data = json.load(f)
            import plotly.graph_objects as go
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"BERT (AUC = {roc_data.get('auc', 0):.3f})",
                line=dict(color="#2563eb", width=2.5),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Chance (AUC = 0.50)",
                line=dict(dash="dash", color="#94a3b8"),
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=350,
                margin=dict(t=20, l=20, r=20, b=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        st.subheader("Precision-Recall Curve")
        if pr_file.exists():
            with open(pr_file, "r", encoding="utf-8") as f:
                pr_data = json.load(f)
            import plotly.graph_objects as go
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_data["recall"],
                y=pr_data["precision"],
                mode="lines",
                name="BERT Precision-Recall",
                line=dict(color="#059669", width=2.5),
            ))
            fig_pr.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=350,
                margin=dict(t=20, l=20, r=20, b=20),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    # 4. Training History Curves
    if history_file.exists():
        st.divider()
        st.subheader("Training Convergence & Loss Progression")
        with open(history_file, "r", encoding="utf-8") as f:
            hist_meta = json.load(f)

        hist_entries = hist_meta.get("history", [])
        if hist_entries:
            import plotly.graph_objects as go
            epochs = [e["epoch"] for e in hist_entries]
            t_loss = [e["train_loss"] for e in hist_entries]
            v_loss = [e["val_loss"] for e in hist_entries]
            v_f1 = [e["val_f1"] for e in hist_entries]

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs, y=t_loss, mode="lines+markers", name="Train Loss", line=dict(color="#ef4444")))
                fig_loss.add_trace(go.Scatter(x=epochs, y=v_loss, mode="lines+markers", name="Val Loss", line=dict(color="#3b82f6")))
                fig_loss.update_layout(
                    title="Train vs Validation Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Cross-Entropy Loss",
                    height=300,
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_h2:
                fig_f1 = go.Figure()
                fig_f1.add_trace(go.Scatter(x=epochs, y=v_f1, mode="lines+markers", name="Validation Macro F1", line=dict(color="#10b981")))
                fig_f1.update_layout(
                    title="Validation F1 Score Progression",
                    xaxis_title="Epoch",
                    yaxis_title="Macro F1",
                    height=300,
                )
                st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown(
            f"**Training Device:** `{hist_meta.get('device_used', 'N/A')}` | "
            f"**Total Run Duration:** `{hist_meta.get('total_duration_seconds', 0)} seconds` | "
            f"**Max Sequence Length:** `{hist_meta.get('max_length', 256)}`"
        )

    # 5. Model Limitations & Critical Interpretation
    st.divider()
    st.subheader("📌 Model Strengths and Practical Limitations")
    st.markdown(
        """
        - **Semantic Grasp:** BERT's bidirectional multi-head attention captures subtle linguistic cues, emotional tone, sensational phrasing, and framing patterns far better than static bag-of-words or TF-IDF models.
        - **Data Leakage Mitigation:** By stripping publisher datelines (such as standard Reuters attribution prefixes found in raw datasets), the model is forced to evaluate actual narrative structure rather than memorizing publisher strings.
        - **Temporal & Domain Drift:** News classifiers trained on historical events (e.g. 2016-2018 politics) experience performance degradation on novel geopolitical events.
        - **Attribution vs Truth:** High confidence indicates stylistic and token similarity to the training corpus's labeled deceptive stories, not an ontological fact check of real-world veracity.
        """
    )
