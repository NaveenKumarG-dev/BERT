"""
Streamlit Page 2: Deep-Dive Model Explainability & Token Attribution.
Provides interactive word importance heatmaps, contribution bar charts, and interpretation guidance.
"""

from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Explainability Deep-Dive | BERT",
    page_icon="💡",
    layout="wide",
)

st.title("💡 Model Explainability & Token Attribution")
st.markdown(
    "Demystify BERT predictions by examining how individual words and tokens sway the classifier's output. "
    "Attribution is computed via gradient-weighted input embeddings, highlighting linguistic signals that increase or decrease the probability of deception."
)

@st.cache_resource(show_spinner="Loading explainability engine...")
def get_cached_explainer():
    from src.explain import NewsExplainer
    model_dir = Path("./artifacts/model").resolve()
    if not (model_dir / "config.json").exists():
        return None
    try:
        return NewsExplainer(model_dir=model_dir)
    except Exception as e:
        st.error(f"Failed to load explainer: {e}")
        return None


explainer = get_cached_explainer()

if explainer is None:
    st.warning("⚠️ Trained model artifact not found in `artifacts/model/`.")
    st.info("Please train the model first by running `python src/train.py`.")
else:
    # Sample pre-fill options
    SAMPLE_OPTIONS = {
        "Custom Input": ("", ""),
        "Sensational Conspiracy (Likely Fake)": (
            "SHOCK CLAIM: Secret underground vaults hold hidden energy technology",
            "Insiders from a clandestine military program have reportedly confirmed the existence of covert facilities holding suppressed energy devices. Mainstream media continues its total blackout on this groundbreaking disclosure."
        ),
        "Official Policy Report (Likely Real)": (
            "Federal Reserve holds interest rates steady amid moderated inflation",
            "Central bank policymakers voted unanimously to maintain the benchmark lending rate following a two-day policy session, noting sustained labor market resilience and gradual progress toward target inflation goals."
        ),
    }

    selected_sample = st.selectbox("Select an illustrative sample or write your own:", list(SAMPLE_OPTIONS.keys()))
    default_h, default_b = SAMPLE_OPTIONS[selected_sample]

    with st.form("explain_form"):
        h_input = st.text_input("Headline:", value=default_h, placeholder="Enter headline...")
        b_input = st.text_area("Article Content:", value=default_b, height=160, placeholder="Enter body text...")
        target_mode = st.radio(
            "Attribution Target:",
            ["Explain Model's Predicted Class (Default)", "Force Explain 'Fake' Class", "Force Explain 'Real' Class"],
            horizontal=True,
        )
        submit_btn = st.form_submit_button("⚡ Compute Attribution", type="primary")

    if submit_btn:
        combined_text = f"{h_input} {b_input}".strip()
        if not combined_text or len(combined_text) < 10:
            st.warning("Please enter at least 10 characters of text to explain.")
        else:
            target_class_id = None
            if target_mode == "Force Explain 'Fake' Class":
                target_class_id = 1
            elif target_mode == "Force Explain 'Real' Class":
                target_class_id = 0

            with st.spinner("Calculating embedding gradients..."):
                try:
                    exp = explainer.explain(headline=h_input, body=b_input, target_class=target_class_id)
                except Exception as e:
                    st.error(f"Error computing explanation: {e}")
                    exp = None

            if exp:
                st.divider()
                st.subheader(f"Attribution Target: {exp['target_class']} News")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Predicted Label", exp["predicted_class"])
                with c2:
                    st.metric("Fake Probability", f"{exp['probabilities']['Fake'] * 100:.1f}%")
                with c3:
                    st.metric("Real Probability", f"{exp['probabilities']['Real'] * 100:.1f}%")

                st.markdown("#### Inline Word Attribution Heatmap")
                st.markdown(
                    "Words colored in <span style='color: #ef4444; font-weight: bold;'>Red</span> pushed toward **Fake**, "
                    "while words in <span style='color: #10b981; font-weight: bold;'>Green</span> pushed toward **Real**.",
                    unsafe_allow_html=True,
                )
                st.markdown(exp["html_markup"], unsafe_allow_html=True)

                st.divider()
                st.subheader("Top Influential Terms")

                col_fake_bar, col_real_bar = st.columns(2)

                top_fake = exp["top_words_for_target"] if exp["target_id"] == 1 else exp["top_words_against_target"]
                top_real = exp["top_words_for_target"] if exp["target_id"] == 0 else exp["top_words_against_target"]

                with col_fake_bar:
                    st.markdown("**Terms Increasing 'Fake' Confidence**")
                    if top_fake:
                        import plotly.graph_objects as go
                        words = [item["word"] for item in top_fake[:10]][::-1]
                        scores = [abs(item["score"]) for item in top_fake[:10]][::-1]
                        fig = go.Figure(go.Bar(
                            x=scores, y=words, orientation='h',
                            marker_color="#ef4444"
                        ))
                        fig.update_layout(height=300, margin=dict(t=10, l=10, r=10, b=10), xaxis_title="Attribution Magnitude")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No terms significantly favored Fake.")

                with col_real_bar:
                    st.markdown("**Terms Increasing 'Real' Confidence**")
                    if top_real:
                        import plotly.graph_objects as go
                        words = [item["word"] for item in top_real[:10]][::-1]
                        scores = [abs(item["score"]) for item in top_real[:10]][::-1]
                        fig = go.Figure(go.Bar(
                            x=scores, y=words, orientation='h',
                            marker_color="#10b981"
                        ))
                        fig.update_layout(height=300, margin=dict(t=10, l=10, r=10, b=10), xaxis_title="Attribution Magnitude")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No terms significantly favored Real.")

                st.divider()
                st.markdown(
                    """
                    ### 🔍 How to Interpret These Attributions
                    - **Attribution vs Truth:** High attribution scores indicate which words the neural network attended to when making its classification. It **does not prove** whether an individual sentence is factually true or false in reality.
                    - **Sensational Vocabulary:** Words with extreme emotional valence (*"shocking"*, *"whistleblower"*, *"suppressed"*, *"clandestine"*) frequently exhibit high positive attribution toward the Fake news label.
                    - **Institutional & Formal Syntax:** Neutral, measured phrasing (*"spokesperson"*, *"unanimously"*, *"announced"*, *"budget"*) tends to generate attribution toward the Real news label.
                    """
                )
