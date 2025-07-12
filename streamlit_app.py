import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lime
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

# ==== Styling und Layout ====
st.set_page_config(
    page_title="🗳️ Tweetseek - Parteivorhersage",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für besseres Styling
st.markdown("""
<style>
    /* Hauptcontainer */
    .main {
        padding-top: 2rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metriken Cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Vorhersage Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    
    /* Eingabefeld Styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Titel Styling */
    .main-title {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    /* Party Colors */
    .party-afd { border-left-color: #009ee0; }
    .party-gruen { border-left-color: #64a12d; }
    .party-cdu { border-left-color: #000000; }
    .party-csu { border-left-color: #0c3c85; }
    .party-spd { border-left-color: #e3000f; }
    .party-fdp { border-left-color: #ffed00; }
    .party-linke { border-left-color: #be3075; }
    .party-fraktionslos { border-left-color: #999999; }
</style>
""", unsafe_allow_html=True)

# ==== Beispiel-Tweets nach Themen ====
SAMPLE_TWEET_CATEGORIES = {
    "Klima & Umwelt": [
        "Klimaschutz muss höchste Priorität haben – für unsere Zukunft.",
        "Wir brauchen eine echte Energiewende, nicht nur leere Versprechen.",
        "Erneuerbare Energien sind der Schlüssel für eine saubere Zukunft."
    ],
    "Migration & Integration": [
        "Grenzen sichern heißt Verantwortung übernehmen.",
        "Integration gelingt nur mit klaren Regeln und Erwartungen.",
        "Wir müssen sowohl humanitär als auch pragmatisch handeln."
    ],
    "Soziales & Gerechtigkeit": [
        "Gerechtigkeit heißt: faire Löhne und sichere Renten.",
        "Das Bürgergeld stärkt den sozialen Zusammenhalt.",
        "Niemand darf in Deutschland in Armut leben müssen."
    ],
    "Wirtschaft & Finanzen": [
        "Wir entlasten den Mittelstand und senken die Steuerlast.",
        "Innovationen und Unternehmertum sind der Schlüssel für Wachstum.",
        "Die soziale Marktwirtschaft ist unser bewährtes Modell."
    ],
    "Digitalisierung & Bildung": [
        "Deutschland braucht flächendeckendes Glasfaser und 5G – jetzt!",
        "Künstliche Intelligenz bietet große Chancen für unsere Wirtschaft.",
        "Bildung ist die wichtigste Investition in unsere Zukunft."
    ]
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

# ==== Titel ====
st.markdown('<h1 class="main-title">🗳️ Tweetseek - Parteivorhersage mit KI</h1>', unsafe_allow_html=True)
st.markdown("---")

# ==== Sidebar: Parteiinformationen ====
with st.sidebar:
    st.markdown("## ℹ️ Parteiinformationen")
    for partei, info in PARTY_INFOS.items():
        with st.expander(f"{partei}", expanded=False):
            st.write(info)
    
    st.markdown("---")
    st.markdown("## 📊 Modell-Info")
    st.info("""
    **Modell**: XGBoost mit kombinierten Features
    - 🔤 TF-IDF Textanalyse
    - 🧠 BERT Embeddings  
    - 📈 Engineered Features
    
    **Genauigkeit**: ~35% bei 8 Parteien
    """)

# ==== Modell laden ====
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/xgb_model_combined.joblib")
        vectorizer = joblib.load("models/tfidf_vectorizer_combined.joblib")
        scaler = joblib.load("models/scaler_combined.joblib")
        label_encoder = joblib.load("models/label_encoder_combined.joblib")
        
        # BERT-Modell laden
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        bert_model = AutoModel.from_pretrained("bert-base-german-cased")
        bert_model.eval()
        
        return model, vectorizer, scaler, label_encoder, tokenizer, bert_model
    except FileNotFoundError as e:
        st.error(f"❌ Fehler beim Laden der Modelldateien: {e}")
        st.error("Stellen Sie sicher, dass alle Modell-Dateien im 'models/' Ordner vorhanden sind.")
        st.stop()

model, vectorizer, scaler, label_encoder, tokenizer, bert_model = load_models()

# ==== Feature-Extraktion ====
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_emojis(text): 
    return sum(1 for char in str(text) if ord(char) > 127 and ord(char) < 4000)

def avg_word_length(text):
    words = re.findall(r"\w+", str(text))
    return sum(len(w) for w in words) / len(words) if words else 0

def uppercase_ratio(text): 
    return sum(1 for c in str(text) if c.isupper()) / len(str(text)) if len(str(text)) > 0 else 0

def multi_punct_count(text): 
    return len(re.findall(r"[!?]{2,}", str(text)))

def count_political_terms(text): 
    return sum(1 for w in POLITICAL_TERMS if w in str(text).lower())

def count_hashtags(text): 
    return len(re.findall(r"#\w+", str(text)))

def count_mentions(text): 
    return len(re.findall(r"@\w+", str(text)))

def count_urls(text): 
    return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))

def count_dots(text): 
    return len(re.findall(r"\.\.+", str(text)))

def is_retweet(text): 
    return int(str(text).strip().lower().startswith("rt @"))

def extract_features(text):
    features = [
        len(str(text)),                    # tweet_length_chars
        len(str(text).split()),            # tweet_length_words  
        avg_word_length(text),             # avg_word_length
        uppercase_ratio(text),             # uppercase_ratio
        str(text).count("!"),              # exclamations
        str(text).count("?"),              # questions
        multi_punct_count(text),           # multi_punct_count
        count_political_terms(text),       # political_term_count
        count_emojis(text),                # num_emojis
        count_hashtags(text),              # num_hashtags
        count_mentions(text),              # num_mentions
        count_urls(text),                  # num_urls
        count_dots(text),                  # dots
        is_retweet(text)                   # is_retweet
    ]
    return np.array(features).reshape(1, -1)

def embed_single_text(text):
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        output = bert_model(**encoded)
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

def predict_with_all_features(text):
    """Vollständige Vorhersage mit allen Features für das Hauptmodell"""
    # TF-IDF
    X_tfidf = vectorizer.transform([text])
    
    # BERT 
    X_bert = embed_single_text(text)
    
    # Engineered Features
    X_eng = extract_features(text)
    X_eng_scaled = scaler.transform(X_eng)
    
    # Kombinieren
    X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
    
    # Vorhersage
    pred_encoded = model.predict(X_all)[0]
    pred = label_encoder.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(X_all)[0]
    
    return pred, probs, X_all

# ==== LIME Setup ====
@st.cache_resource
def setup_lime():
    """LIME Explainer initialisieren"""
    explainer = LimeTextExplainer(
        class_names=label_encoder.classes_,
        feature_selection='auto',
        verbose=False,
        mode='classification'
    )
    return explainer

def lime_predict_fn(texts):
    """Wrapper-Funktion für LIME - muss Liste von Texten verarbeiten"""
    results = []
    for text in texts:
        try:
            _, probs, _ = predict_with_all_features(text)
            results.append(probs)
        except Exception as e:
            # Fallback bei Fehlern
            results.append(np.ones(len(label_encoder.classes_)) / len(label_encoder.classes_))
    return np.array(results)

def get_lime_explanation(text, num_features=10):
    """LIME Erklärung für einen Text generieren"""
    explainer = setup_lime()
    
    with st.spinner("🔍 Analysiere Worteinflüsse mit LIME..."):
        explanation = explainer.explain_instance(
            text, 
            lime_predict_fn, 
            num_features=num_features,
            num_samples=500  # Reduziert für bessere Performance
        )
    
    return explanation

# ==== UI: Hauptbereich ====
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Tweet eingeben")
    
    # Thema-Auswahl
    selected_theme = st.selectbox(
        "🎯 Wähle ein Thema für Beispiele:",
        list(SAMPLE_TWEET_CATEGORIES.keys()),
        help="Wähle ein politisches Thema, um passende Beispiel-Tweets zu laden"
    )

with col2:
    st.markdown("### 🎲 Beispiel laden")
    if st.button("🔄 Zufälliger Beispiel-Tweet", help="Lädt einen zufälligen Tweet aus der gewählten Kategorie"):
        if "input_tweet" not in st.session_state:
            st.session_state["input_tweet"] = ""
        st.session_state["input_tweet"] = random.choice(SAMPLE_TWEET_CATEGORIES[selected_theme])

# Haupteingabefeld
if "input_tweet" not in st.session_state:
    st.session_state["input_tweet"] = ""

current_tweet_input = st.text_area(
    "Gib hier deinen Tweet ein:",
    placeholder="Beispiel: Klimaschutz ist die wichtigste Aufgabe unserer Zeit. Wir müssen jetzt handeln! #Klimawandel",
    height=120,
    value=st.session_state.get("input_tweet", ""),
    help="Gib einen deutschen politischen Tweet ein (max. 280 Zeichen)"
)

st.session_state["input_tweet"] = current_tweet_input

# Analyse-Buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_clicked = st.button("🔮 Vorhersage starten", help="Startet die Parteivorhersage für den eingegebenen Tweet")

with col2:
    explain_clicked = st.button("🔍 Mit LIME erklären", help="Zeigt an, welche Wörter die Entscheidung beeinflusst haben")

with col3:
    both_clicked = st.button("🚀 Beides ausführen", help="Führt sowohl Vorhersage als auch Erklärung aus")

# ==== Hauptanalyse ====
if (predict_clicked or both_clicked or explain_clicked) and st.session_state["input_tweet"].strip():
    tweet_to_predict = st.session_state["input_tweet"].strip()
    
    # Eingabe-Validierung
    if len(tweet_to_predict) > 500:
        st.warning("⚠️ Der Tweet ist sehr lang. Für bessere Ergebnisse sollte er kürzer als 280 Zeichen sein.")
    
    if len(tweet_to_predict) < 10:
        st.warning("⚠️ Der Tweet ist sehr kurz. Längere Texte liefern meist bessere Vorhersagen.")
    
    # Vorhersage ausführen
    if predict_clicked or both_clicked:
        with st.spinner("🤖 Analysiere Tweet..."):
            pred, probs, X_all = predict_with_all_features(tweet_to_predict)
        
        # ==== Ergebnisse anzeigen ====
        st.markdown("---")
        st.markdown("## 🎯 Vorhersage-Ergebnisse")
        
        # Hauptvorhersage prominent anzeigen
        confidence = max(probs)
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🏆 Vorhergesagte Partei: {pred}</h2>
            <h3>Konfidenz: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Wahrscheinlichkeitsverteilung
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📊 Wahrscheinlichkeitsverteilung")
            
            # Sortierte Daten für bessere Visualisierung
            parties = label_encoder.inverse_transform(range(len(probs)))
            df_probs = pd.DataFrame({
                "Partei": parties,
                "Wahrscheinlichkeit": probs,
                "Farbe": [PARTY_COLORS.get(p, "#cccccc") for p in parties]
            }).sort_values("Wahrscheinlichkeit", ascending=False)
            
            # Interaktives Plotly Chart
            fig = px.bar(
                df_probs, 
                x="Wahrscheinlichkeit", 
                y="Partei",
                orientation='h',
                color="Farbe",
                color_discrete_map="identity",
                title="Vorhersagewahrscheinlichkeiten"
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Wahrscheinlichkeit",
                yaxis_title="Partei"
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Wahrscheinlichkeit: %{x:.2%}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏛️ Top 3 Parteien")
            for i, (_, row) in enumerate(df_probs.head(3).iterrows()):
                rank_emoji = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{rank_emoji} {row['Partei']}</h4>
                    <h3 style="color: {row['Farbe']};">{row['Wahrscheinlichkeit']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Partei-Info anzeigen
            with st.expander(f"ℹ️ Mehr über {pred}"):
                st.write(PARTY_INFOS.get(pred, "Keine Informationen verfügbar."))
    
    # LIME Erklärung
    if explain_clicked or both_clicked:
        st.markdown("---")
        st.markdown("## 🔍 LIME-Erklärung: Worteinflüsse")
        
        try:
            explanation = get_lime_explanation(tweet_to_predict, num_features=10)
            
            # LIME Ergebnisse verarbeiten
            word_weights = explanation.as_list()
            predicted_class = explanation.available_labels()[0]
            predicted_party = label_encoder.inverse_transform([predicted_class])[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 📝 Worteinflüsse für Vorhersage: **{predicted_party}**")
                
                # Wort-Gewichte Tabelle
                if word_weights:
                    df_words = pd.DataFrame(word_weights, columns=["Wort", "Gewicht"])
                    df_words["Einfluss"] = df_words["Gewicht"].apply(
                        lambda x: "🟢 Positiv" if x > 0 else "🔴 Negativ"
                    )
                    df_words["Stärke"] = abs(df_words["Gewicht"])
                    
                    # Sortiert nach absoluter Stärke
                    df_words = df_words.sort_values("Stärke", ascending=False)
                    
                    st.dataframe(
                        df_words[["Wort", "Einfluss", "Gewicht"]].round(3),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualisierung der Wort-Gewichte
                    fig_words = px.bar(
                        df_words.head(8),
                        x="Gewicht",
                        y="Wort",
                        orientation='h',
                        color="Gewicht",
                        color_continuous_scale="RdYlGn",
                        title="Wort-Einflüsse (LIME)"
                    )
                    fig_words.update_layout(height=400)
                    st.plotly_chart(fig_words, use_container_width=True)
                else:
                    st.warning("⚠️ Keine aussagekräftigen Wort-Einflüsse gefunden.")
            
            with col2:
                st.markdown("### ℹ️ LIME-Erklärung")
                st.info("""
                **LIME** (Local Interpretable Model-agnostic Explanations) zeigt:
                
                🟢 **Positive Werte**: Wörter, die für diese Partei sprechen
                
                🔴 **Negative Werte**: Wörter, die gegen diese Partei sprechen
                
                📊 **Stärke**: Je größer der Betrag, desto wichtiger das Wort
                """)
                
                # Performance-Hinweis
                st.warning("""
                ⏱️ **Hinweis**: LIME-Analysen dauern länger, da das Modell hunderte Text-Variationen analysiert.
                """)
        
        except Exception as e:
            st.error(f"❌ Fehler bei LIME-Analyse: {str(e)}")
            st.info("💡 Tipp: Versuche es mit einem anderen Tweet oder prüfe die Modell-Dateien.")

# ==== Tweet-Analyse Metriken ====
if st.session_state["input_tweet"].strip():
    st.markdown("---")
    st.markdown("## 📈 Tweet-Eigenschaften")
    
    tweet_text = st.session_state["input_tweet"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Zeichen", 
            len(tweet_text),
            help="Gesamtanzahl der Zeichen im Tweet"
        )
        st.metric(
            "💬 Wörter", 
            len(tweet_text.split()),
            help="Anzahl der Wörter im Tweet"
        )
    
    with col2:
        st.metric(
            "#️⃣ Hashtags", 
            count_hashtags(tweet_text),
            help="Anzahl der Hashtags (#) im Tweet"
        )
        st.metric(
            "👤 Mentions", 
            count_mentions(tweet_text),
            help="Anzahl der Erwähnungen (@) im Tweet"
        )
    
    with col3:
        st.metric(
            "🔗 URLs", 
            count_urls(tweet_text),
            help="Anzahl der Links im Tweet"
        )
        st.metric(
            "😀 Emojis", 
            count_emojis(tweet_text),
            help="Anzahl der Emojis im Tweet"
        )
    
    with col4:
        st.metric(
            "🏛️ Politik-Begriffe", 
            count_political_terms(tweet_text),
            help="Anzahl erkannter politischer Schlüsselwörter"
        )
        retweet_status = "Ja" if is_retweet(tweet_text) else "Nein"
        st.metric(
            "🔄 Retweet", 
            retweet_status,
            help="Ist dieser Tweet ein Retweet?"
        )

elif predict_clicked or explain_clicked or both_clicked:
    st.warning("⚠️ Bitte gib zuerst einen Tweet ein!")

# ==== Footer ====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 <strong>Tweetseek</strong> - KI-basierte Parteivorhersage für deutsche Tweets</p>
    <p>Entwickelt mit Streamlit, XGBoost, BERT und LIME</p>
    <p><em>Hinweis: Diese Vorhersagen dienen nur zu Demonstrationszwecken und spiegeln nicht die tatsächlichen politischen Ansichten wider.</em></p>
</div>
""", unsafe_allow_html=True)