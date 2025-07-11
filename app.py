import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel

# ==== 模型配置 ====
MODEL_OPTIONS = {
    "TF-IDF baseline": {
        "model": "models/lr_model_no_urls.joblib",
        "vectorizer": "models/tfidf_no_urls.joblib",
        "scaler": None
    },
    "TF-IDF + Extra Features": {
        "model": "models/lr_model_extra_no_urls.joblib",
        "vectorizer": "models/tfidf_extra_no_urls.joblib",
        "scaler": "models/scaler_extra_no_urls.joblib"
    },
    "TF-IDF + BERT + Engineered": {
        "model": "models/lr_tfidf_bert_engineered.joblib",
        "vectorizer": "models/tfidf_vectorizer_bert_engineered.joblib",
        "scaler": "models/feature_scaler_bert_engineered.joblib"
    }
}

st.set_page_config(page_title="Parteivorhersage", layout="centered")

st.title("🗳️ Parteivorhersage für Bundestags-Tweets")
choice = st.selectbox("🔍 Wähle ein Modell:", list(MODEL_OPTIONS.keys()))
info = MODEL_OPTIONS[choice]

model = joblib.load(info["model"])
vectorizer = joblib.load(info["vectorizer"])
scaler = joblib.load(info["scaler"]) if info["scaler"] else None
use_bert = "BERT" in choice

# ==== BERT vorbereiten ====
if use_bert:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    bert_model = AutoModel.from_pretrained("bert-base-german-cased")
    bert_model.eval()

# ==== Text Feature Funktionen ====
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_emojis(text): return str(text).count("😀")  # placeholder
def avg_word_length(text):
    words = re.findall(r"\w+", str(text))
    return sum(len(w) for w in words) / len(words) if words else 0

def uppercase_ratio(text):
    text = str(text)
    return sum(1 for c in text if c.isupper()) / len(text) if text else 0

def multi_punct_count(text): return len(re.findall(r"[!?]{2,}", str(text)))
def count_political_terms(text): return sum(1 for w in POLITICAL_TERMS if w in str(text).lower())
def count_hashtags(text): return len(re.findall(r"#\w+", str(text)))
def count_mentions(text): return len(re.findall(r"@\w+", str(text)))
def count_urls(text): return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))
def count_dots(text): return len(re.findall(r"\.\.+", str(text)))
def is_retweet(text): return int(str(text).strip().lower().startswith("rt @"))

def extract_features(text):
    return np.array([[
        len(str(text)),
        len(str(text).split()),
        avg_word_length(text),
        uppercase_ratio(text),
        str(text).count("!"),
        str(text).count("?"),
        multi_punct_count(text),
        count_political_terms(text),
        count_emojis(text),
        count_hashtags(text),
        count_mentions(text),
        count_urls(text),
        count_dots(text),
        is_retweet(text),
    ]])

def extract_extra_features(text):
    return np.array([[
        count_emojis(text),
        count_hashtags(text),
        count_mentions(text),
        count_urls(text)
    ]])

def embed_single_text(text):
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        output = bert_model(**encoded)
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

# ==== UI ====
tweet = st.text_area("📝 Gib einen Bundestags-Tweet ein:", height=100)

if tweet and st.button("🔮 Vorhersagen"):
    X_tfidf = vectorizer.transform([tweet])

    X_eng_scaled = None
    if scaler:
        if "Extra Features" in choice:
            X_eng = extract_extra_features(tweet)
        else:
            X_eng = extract_features(tweet)
        X_eng_scaled = scaler.transform(X_eng)

    if use_bert:
        X_bert = embed_single_text(tweet)
        X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
    elif scaler:
        X_all = np.hstack([X_tfidf.toarray(), X_eng_scaled])
    else:
        X_all = X_tfidf

    pred = model.predict(X_all)[0]
    st.success(f"🗳️ Vorhergesagte Partei: **{pred}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_all)[0]
        st.subheader("📊 Vorhersagewahrscheinlichkeit")
        st.bar_chart({p: float(prob) for p, prob in zip(model.classes_, probs)})

st.markdown("---")
st.caption("📌 Dieses Tool wurde im Rahmen des ML4B-Projekts entwickelt – zur Parteivorhersage deutscher Bundestags-Tweets.")
