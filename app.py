
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "waste_items.csv"

st.set_page_config(page_title="Paras", page_icon="♻️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(data):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(data["item"], data["category"])
    return model

def find_best_record(text, data):
    text = text.lower().strip()
    exact = data[data["item"].str.lower() == text]
    if not exact.empty:
        return exact.iloc[0]

    # Lightweight keyword/substring matching improves guidance for phrases
    # while the ML model provides the category prediction.
    scores = []
    for _, row in data.iterrows():
        terms = set(re.findall(r"[a-z0-9]+", row["item"].lower()))
        tokens = set(re.findall(r"[a-z0-9]+", text))
        scores.append((len(terms & tokens), row))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1] if scores and scores[0][0] > 0 else None

def ai_advice(text, predicted, record):
    # Optional generative-AI layer. The core application remains usable offline.
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"""You are Paras, a sustainability assistant.
Waste description: {text}
Predicted category: {predicted}
Known material: {record['material'] if record is not None else 'unknown'}
Give concise, practical disposal guidance. Do not invent local rules.
Clearly say that local waste authority guidance may differ."""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception:
            pass
    return None

data = load_data()
model = train_model(data)

st.markdown("""
<style>
.main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 1.05rem; color: #555; margin-top: 0.2rem;}
.card {padding: 1rem; border-radius: 14px; border: 1px solid #ddd; background: #fafafa;}
</style>
""", unsafe_allow_html=True)

st.title("♻️ Paras")
st.markdown("### AI-Powered Smart Waste Segregation & Recycling Assistant")
st.write("Describe an everyday waste item and get an AI/ML-supported category, disposal guidance, and sustainability tip.")

with st.sidebar:
    st.header("About the project")
    st.write("Paras supports SDG 12: Responsible Consumption and Production.")
    st.info("Prototype note: disposal rules vary by location. Always verify special/hazardous waste instructions with your local authority.")
    st.caption("Core model: TF-IDF + Logistic Regression")
    st.caption("Optional AI: Gemini API via GEMINI_API_KEY")

tab1, tab2, tab3 = st.tabs(["🔎 Analyze Waste", "📊 Dataset Insights", "ℹ️ Responsible AI"])

with tab1:
    examples = ["Old mobile phone", "Banana peel", "Plastic bottle", "Used battery", "Cardboard box"]
    selected = st.selectbox("Try an example", ["— Select —"] + examples)
    default = "" if selected == "— Select —" else selected
    query = st.text_input("What waste item do you have?", value=default, placeholder="e.g., old mobile phone")

    if st.button("Analyze ♻️", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a waste item.")
        else:
            predicted = model.predict([query])[0]
            record = find_best_record(query, data)

            st.subheader("Analysis Result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Category", predicted)
            c2.metric("Material", record["material"] if record is not None else "Unknown")
            c3.metric("Guidance Type", record["category"] if record is not None else "Check Local Guidance")

            if record is not None:
                st.success(f"**Recommended action:** {record['disposal_guidance']}")
                st.info(f"🌱 **Eco tip:** {record['eco_tip']}")
            else:
                st.warning("The item is outside the prototype examples. Use local waste-authority guidance before disposal.")

            gen = ai_advice(query, predicted, record)
            if gen:
                st.markdown("#### 🤖 Generative AI Guidance")
                st.write(gen)

            st.caption("This prototype provides informational decision support; it is not a substitute for official local waste-management instructions.")

with tab2:
    st.subheader("Prototype Dataset")
    st.dataframe(data, use_container_width=True, hide_index=True)
    counts = data["category"].value_counts()
    st.bar_chart(counts)
    st.write(f"Dataset records: **{len(data)}**")

with tab3:
    st.subheader("Responsible AI Considerations")
    st.markdown("""
- **Fairness:** The prototype uses general waste descriptions and does not use personal characteristics.
- **Transparency:** The category is produced by a TF-IDF + Logistic Regression classifier trained on the included sample dataset.
- **Ethics:** The system avoids presenting uncertain disposal rules as universal facts.
- **Privacy:** No personal information is required to analyze an item.
- **Safety:** Hazardous and e-waste items are flagged for appropriate/special handling rather than ordinary disposal.
- **Local variation:** Recycling acceptance differs by municipality, so users should verify official local instructions.
""")
