import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import random

# ==== Beispiel-Tweets nach Themen ====
SAMPLE_TWEET_CATEGORIES = {
    "Klima": [
        "Klimaschutz muss höchste Priorität haben – für unsere Zukunft.",
        "Wir brauchen eine echte Energiewende, nicht nur leere Versprechen."
    ],
    "Migration": [
        "Grenzen sichern heißt Verantwortung übernehmen.",
        "Integration gelingt nur mit klaren Regeln und Erwartungen."
    ],
    "Soziales": [
        "Gerechtigkeit heißt: faire Löhne und sichere Renten.",
        "Das Bürgergeld stärkt den sozialen Zusammenhalt."
    ],
    "Wirtschaft": [
        "Wir entlasten den Mittelstand und senken die Steuerlast.",
        "Innovationen und Unternehmertum sind der Schlüssel für Wachstum."
    ],
    "Digitales": [
        "Deutschland braucht flächendeckendes Glasfaser und 5G – jetzt!",
        "Künstliche Intelligenz bietet große Chancen für unsere Wirtschaft."
    ],
    "Bildung": [
        "Bildung darf nicht vom Geldbeutel der Eltern abhängen.",
        "Mehr Lehrer, bessere Ausstattung – wir investieren in die Zukunft."
    ],
    "Europa": [
        "Ein starkes Europa ist unser Garant für Frieden und Wohlstand.",
        "Wir stehen zu unserer Verantwortung in der EU."
    ],
    "Sicherheit": [
        "Mehr Mittel für Polizei und Justiz – für Ihre Sicherheit.",
        "Wir stärken die Bundeswehr und unsere Verteidigungsfähigkeit."
    ],
    "Freiheit": [
        "Freiheit und Grundrechte sind nicht verhandelbar.",
        "Wir setzen uns gegen jede Form der Zensur ein."
    ],
    "Gesundheit": [
        "Pflegekräfte verdienen mehr Wertschätzung – und bessere Löhne.",
        "Ein stabiles Gesundheitssystem ist keine Selbstverständlichkeit."
    ]
}

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
    # 仅在需要时加载BERT模型和tokenizer，避免不必要的内存占用
    @st.cache_resource
    def load_bert_model():
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        bert_model = AutoModel.from_pretrained("bert-base-german-cased")
        bert_model.eval()
        return tokenizer, bert_model
    tokenizer, bert_model = load_bert_model()

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

# ==== UI: Text + Thema + Buttons ====
# 确保 session_state['input_tweet'] 在使用前被初始化
if "input_tweet" not in st.session_state:
    st.session_state["input_tweet"] = "" # 初始值为空字符串

thema = st.selectbox("📂 Wähle ein Thema:", list(SAMPLE_TWEET_CATEGORIES.keys()))

col1, col2 = st.columns([3, 1])
with col1:
    # 文本区域的key直接绑定到session_state，并且初始值从session_state获取
    current_tweet_input = st.text_area(
        label="",
        placeholder="Gib einen Bundestags-Tweet ein...",
        height=100,
        label_visibility="collapsed",
        value=st.session_state["input_tweet"], # 使用 session_state 的值作为初始值
        key="input_tweet_widget" # 给文本区域一个不同的key，避免和session_state的key混淆
    )
    # 如果用户直接在文本框输入，更新 session_state
    st.session_state["input_tweet"] = current_tweet_input

with col2:
    if st.button("🔄 Beispiel-Tweet laden"):
        # 当按钮点击时，更新 session_state 中的 input_tweet
        st.session_state["input_tweet"] = random.choice(SAMPLE_TWEET_CATEGORIES[thema])
        # Streamlit 会自动重新运行脚本以反映 session_state 的变化

predict_clicked = st.button("🔮 Vorhersagen")

if predict_clicked and st.session_state["input_tweet"].strip(): # 使用 session_state['input_tweet'] 进行预测
    tweet_to_predict = st.session_state["input_tweet"]

    X_tfidf = vectorizer.transform([tweet_to_predict])

    X_eng_scaled = None
    if scaler:
        X_eng = extract_extra_features(tweet_to_predict) if "Extra Features" in choice else extract_features(tweet_to_predict)
        X_eng_scaled = scaler.transform(X_eng)

    if use_bert:
        X_bert = embed_single_text(tweet_to_predict)
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
        # 创建一个包含概率、类别和颜色的DataFrame
        df = pd.DataFrame({
            "Partei": model.classes_,
            "Wahrscheinlichkeit": probs,
            "Farbe": [PARTY_COLORS.get(p, "#aaaaaa") for p in model.classes_]
        })
        # 按照概率降序排序，使图表更清晰
        df = df.sort_values(by="Wahrscheinlichkeit", ascending=False)


        st.subheader("📊 Vorhersagewahrscheinlichkeit")
        # Streamlit的bar_chart不支持直接传入颜色，所以如果需要自定义颜色，需要使用plotly或其他库。
        # 这里为了简化，直接使用Streamlit内置的bar_chart
        st.bar_chart(data=df.set_index("Partei")["Wahrscheinlichkeit"])

st.markdown("---")
st.caption("📌 Dieses Tool wurde im Rahmen des ML4B-Projekts entwickelt – zur Parteivorhersage deutscher Bundestags-Tweets.")
