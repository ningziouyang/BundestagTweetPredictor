import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# ==== Modelloptionen ====
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

# ==== Partei-Infos & Farben ====
PARTY_INFOS = {
    "AfD": "Alternative für Deutschland: Kritisch gegenüber Migration und EU, betont nationale Interessen und innere Sicherheit.",
    "Bündnis 90/Die Grünen": "Fokus auf Klimaschutz, Nachhaltigkeit, soziale Gerechtigkeit und pro-europäische Zusammenarbeit.",
    "CDU": "Christlich Demokratische Union: Mitte-rechts, wirtschaftsliberal, konservativ in Gesellschaftsfragen.",
    "CSU": "Christlich-Soziale Union: Bayerische Schwesterpartei der CDU, konservativ, betont regionale Identität.",
    "SPD": "Sozialdemokratische Partei Deutschlands: Mitte-links, soziale Gerechtigkeit, Arbeitnehmerrechte, Bildung, Klima.",
    "FDP": "Freie Demokratische Partei: Wirtschaftsliberal, betont individuelle Freiheit, Digitalisierung, Bildung.",
    "Die Linke": "Sozialistisch, betont soziale Gerechtigkeit, Friedenspolitik, starke Regulierung von Wirtschaft.",
    "Fraktionslos": "Keiner Fraktion zugeordnet – meist Einzelabgeordnete oder kleine Gruppen ohne klare Parteizugehörigkeit."
}

PARTY_COLORS = {
    "AfD": "#009ee0",
    "Bündnis 90/Die Grünen": "#64a12d",
    "CDU": "#000000",
    "CSU": "#0c3c85",
    "SPD": "#e3000f",
    "FDP": "#ffed00",
    "Die Linke": "#be3075",
    "Fraktionslos": "#999999"
}

# ==== Layout ====
st.set_page_config(page_title="Parteivorhersage", layout="wide")
st.markdown("""
    <style>
    textarea + div[role='button'] {
        display: none !important;
    }
    .element-container textarea {
        font-size: 16px !important;
    }
    div[data-testid=stTextArea] label {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🗳️ Parteivorhersage für Bundestags-Tweets")

# ==== Seitenleiste: Parteiinformationen ====
st.sidebar.header("ℹ️ Parteiinformationen")
for partei, info in PARTY_INFOS.items():
    with st.sidebar.expander(partei):
        st.write(info)

# ==== Modellauswahl & Laden ====
choice = st.selectbox("🔍 Wähle ein Modell:", list(MODEL_OPTIONS.keys()))
info = MODEL_OPTIONS[choice]

model = joblib.load(info["model"])
vectorizer = joblib.load(info["vectorizer"])
scaler = joblib.load(info["scaler"]) if info["scaler"] else None
use_bert = "BERT" in choice

if use_bert:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    bert_model = AutoModel.from_pretrained("bert-base-german-cased")
    bert_model.eval()

# ==== Feature-Extraktion ====
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_emojis(text): return str(text).count("😀")
def avg_word_length(text):
    words = re.findall(r"\w+", str(text))
    return sum(len(w) for w in words) / len(words) if words else 0
def uppercase_ratio(text): return sum(1 for c in text if c.isupper()) / len(text) if text else 0
def multi_punct_count(text): return len(re.findall(r"[!?]{2,}", str(text)))
def count_political_terms(text): return sum(1 for w in POLITICAL_TERMS if w in str(text).lower())
def count_hashtags(text): return len(re.findall(r"#\w+", str(text)))
def count_mentions(text): return len(re.findall(r"@\w+", str(text)))
def count_urls(text): return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))
def count_dots(text): return len(re.findall(r"\.\.+", str(text)))
def is_retweet(text): return int(str(text).strip().lower().startswith("rt @"))

def extract_features(text):
    return np.array([[len(str(text)), len(str(text).split()), avg_word_length(text), uppercase_ratio(text),
                      str(text).count("!"), str(text).count("?"), multi_punct_count(text), count_political_terms(text),
                      count_emojis(text), count_hashtags(text), count_mentions(text), count_urls(text),
                      count_dots(text), is_retweet(text)]])

def extract_extra_features(text):
    return np.array([[count_emojis(text), count_hashtags(text), count_mentions(text), count_urls(text)]])

def embed_single_text(text):
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        output = bert_model(**encoded)
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

# ==== UI ====
tweet = st.text_area(
    label="",
    placeholder="Gib einen Bundestags-Tweet ein...",
    height=100,
    label_visibility="collapsed",
    key="tweet_input"
)

predict_clicked = st.button("🔮 Vorhersagen")

if predict_clicked and tweet.strip():
    X_tfidf = vectorizer.transform([tweet])

    X_eng_scaled = None
    if scaler:
        X_eng = extract_extra_features(tweet) if "Extra Features" in choice else extract_features(tweet)
        X_eng_scaled = scaler.transform(X_eng)

    if use_bert:
        X_bert = embed_single_text(tweet)
        X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
    elif scaler:
        X_all = np.hstack([X_tfidf.toarray(), X_eng_scaled])
    else:
        X_all = X_tfidf

    pred = model.predict(X_all)[0]
    st.success(f"🟩 Vorhergesagte Partei: **{pred}**")

    with st.expander(f"🧭 Informationen über {pred}"):
        st.write(PARTY_INFOS.get(pred, "Keine Informationen verfügbar."))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_all)[0]
        df = pd.DataFrame({
            "Partei": model.classes_,
            "Wahrscheinlichkeit": probs,
            "Farbe": [PARTY_COLORS.get(p, "#aaaaaa") for p in model.classes_]
        })

        st.subheader("📊 Vorhersagewahrscheinlichkeit")
        st.bar_chart(data=df.set_index("Partei")["Wahrscheinlichkeit"])

st.markdown("---")
st.caption("📌 Dieses Tool wurde im Rahmen des ML4B-Projekts entwickelt – zur Parteivorhersage deutscher Bundestags-Tweets.")
