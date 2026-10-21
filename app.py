"""
Streamlit Web Application: Fake News Detection with BERT and Explainability.
Main page: Interactive Article Classifier with Inline Word Attribution.
"""

from pathlib import Path
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Fake News Detector | BERT",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for a polished, modern interface
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1e293b;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .card-metric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-fake {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        border: 1px solid #fca5a5;
    }
    .badge-real {
        display: inline-block;
        background-color: #d1fae5;
        color: #047857;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        border: 1px solid #6ee7b7;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        color: #64748b;
        background-color: #f1f5f9;
        border-left: 4px solid #94a3b8;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading fine-tuned BERT model and tokenizer...")
def get_cached_predictor():
    """Caches the predictor in memory across app interactions."""
    from src.predict import NewsPredictor
    model_dir = Path("./artifacts/model").resolve()
    if not (model_dir / "config.json").exists():
        return None
    try:
        return NewsPredictor(model_dir=model_dir)
    except Exception as e:
        st.error(f"Failed to load model from {model_dir}: {e}")
        return None


@st.cache_resource(show_spinner="Initializing explainability engine...")
def get_cached_explainer():
    """Caches the explainer in memory."""
    from src.explain import NewsExplainer
    model_dir = Path("./artifacts/model").resolve()
    if not (model_dir / "config.json").exists():
        return None
    try:
        return NewsExplainer(model_dir=model_dir)
    except Exception:
        return None


# Sidebar controls & information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.markdown("### 📰 Navigation & Settings")
    st.markdown(
        "Fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** "
        "trained to distinguish deceptive misinformation from verified reporting."
    )
    st.divider()
    st.markdown("**Model Architecture:** `bert-base-uncased`")
    st.markdown("**Explainability:** Embedding Gradient Attribution")
    st.divider()

    # Pre-canned real-world sample articles
    SAMPLE_REAL = {
        "headline": "Senate reaches bipartisan agreement on infrastructure spending targets",
        "body": "Lawmakers from both parties announced a compromise framework on federal transportation funding, allocating sixty-five billion dollars for broadband network expansion and rail improvements over the next five fiscal years.",
    }
    SAMPLE_FAKE = {
        "headline": "Secret government bunker discovered containing alien technology and suppressed cures",
        "body": "An anonymous whistleblower from an undisclosed agency has revealed explosive documents exposing a massive underground facility where global elites have hidden miraculous cures and covert technology for decades.",
    }

    st.markdown("#### ⚡ Try Sample Articles")
    sample_choice = st.selectbox(
        "Load demonstration text:",
        ["None (Custom Input)", "Sample Real News", "Sample Fake News"],
    )

# Page Header
st.markdown('<div class="main-title">Fake News Detection with BERT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Evaluate article veracity using transformer language representations with token-level explainability.</div>',
    unsafe_allow_html=True,
)

# Populate input fields based on sample selection
initial_headline = ""
initial_body = ""
if sample_choice == "Sample Real News":
    initial_headline = SAMPLE_REAL["headline"]
    initial_body = SAMPLE_REAL["body"]
elif sample_choice == "Sample Fake News":
    initial_headline = SAMPLE_FAKE["headline"]
    initial_body = SAMPLE_FAKE["body"]

# Input form
input_mode = st.radio(
    "Input Layout:",
    ["Separate Headline & Article Body", "Combined Article Text"],
    horizontal=True,
)

if input_mode == "Separate Headline & Article Body":
    headline_input = st.text_input(
        "Article Headline / Title:",
        value=initial_headline,
        placeholder="Enter headline...",
    )
    body_input = st.text_area(
        "Article Body:",
        value=initial_body,
        height=180,
        placeholder="Paste full article text here...",
    )
else:
    combined_val = (
        f"{initial_headline}. {initial_body}".strip()
        if (initial_headline or initial_body)
        else ""
    )
    headline_input = ""
    body_input = st.text_area(
        "Complete News Text:",
        value=combined_val,
        height=220,
        placeholder="Enter headline and article text together...",
    )

col_btn, col_chk = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
with col_chk:
    show_explain = st.checkbox("Generate Word-Level Attribution (Explainability)", value=True)

if analyze_btn:
    text_to_check = f"{headline_input} {body_input}".strip()
    if not text_to_check or len(text_to_check) < 10:
        st.warning("⚠️ Please provide a meaningful headline or article text (at least 10 characters).")
    else:
        predictor = get_cached_predictor()

        if predictor is None:
            st.error(
                "❌ Trained model artifact not found in `artifacts/model/`.\n\n"
                "Please run model training first using:\n"
                "```bash\n"
                "python src/train.py\n"
                "```"
            )
        else:
            with st.spinner("Analyzing text with fine-tuned BERT..."):
                try:
                    result = predictor.predict(headline=headline_input, body=body_input)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    result = None

            if result:
                st.subheader("Classification Result")
                pred_label = result["prediction"]
                conf = result["confidence"]
                p_real = result["probabilities"]["Real"]
                p_fake = result["probabilities"]["Fake"]

                col_res1, col_res2, col_res3 = st.columns([1.2, 1, 1])

                with col_res1:
                    badge_class = "badge-fake" if pred_label == "Fake" else "badge-real"
                    st.markdown(
                        f"""
                        <div style="margin-top: 10px;">
                            <span class="{badge_class}">Predicted: {pred_label.upper()} NEWS</span>
                            <div style="font-size: 0.95rem; color: #475569; margin-top: 8px;">
                                Model Confidence: <strong>{conf * 100:.1f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_res2:
                    st.metric(label="Real News Probability", value=f"{p_real * 100:.1f}%")
                    st.progress(p_real)

                with col_res3:
                    st.metric(label="Fake News Probability", value=f"{p_fake * 100:.1f}%")
                    st.progress(p_fake)

                # Explainability Section
                if show_explain:
                    st.divider()
                    st.subheader("💡 Prediction Explainability (Attribution Map)")
                    st.markdown(
                        "Word-level attribution computed via embedding gradients. "
                        "<span style='color: #ef4444; font-weight: 600;'>Red highlights</span> indicate words that pushed the prediction toward **Fake**, while "
                        "<span style='color: #10b981; font-weight: 600;'>Green highlights</span> pushed toward **Real**.",
                        unsafe_allow_html=True,
                    )

                    explainer = get_cached_explainer()
                    if explainer:
                        with st.spinner("Computing token gradients..."):
                            try:
                                exp_res = explainer.explain(
                                    headline=headline_input,
                                    body=body_input,
                                    target_class=result["label_id"],
                                )
                                st.markdown(exp_res["html_markup"], unsafe_allow_html=True)

                                # Top influential words table
                                col_w1, col_w2 = st.columns(2)
                                with col_w1:
                                    st.markdown("**Top Words Favoring 'Fake' Classification:**")
                                    fake_words = exp_res["top_words_for_target"] if result["label_id"] == 1 else exp_res["top_words_against_target"]
                                    if fake_words:
                                        st.write(", ".join([f"`{item['word']}` ({item['score']:+.2f})" for item in fake_words[:8]]))
                                    else:
                                        st.write("None significantly influential.")

                                with col_w2:
                                    st.markdown("**Top Words Favoring 'Real' Classification:**")
                                    real_words = exp_res["top_words_for_target"] if result["label_id"] == 0 else exp_res["top_words_against_target"]
                                    if real_words:
                                        st.write(", ".join([f"`{item['word']}` ({item['score']:+.2f})" for item in real_words[:8]]))
                                    else:
                                        st.write("None significantly influential.")

                            except Exception as ex:
                                st.warning(f"Could not compute explanation: {ex}")

# Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Classifier Disclaimer:</strong> This application relies on a statistical transformer model fine-tuned on historical news datasets.
        Predictions and word attributions reflect stylistic and linguistic correlations learned during training, and do <strong>not</strong> constitute definitive factual verification.
        Always verify claims using authoritative primary sources and professional fact-checking organizations.
    </div>
    """,
    unsafe_allow_html=True,
)
