import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import random
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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

# ==== Layout-Konfiguration ====
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
    .css-18e3th9 {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗳️ Tweetseek - Parteizugehörigkeiten zuordnen")

# ==== Seitenleiste: Parteiinformationen ====
st.sidebar.header("ℹ️ Parteiinformationen")
for partei, info in PARTY_INFOS.items():
    with st.sidebar.expander(partei):
        st.write(info)

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
        st.error(f"Fehler beim Laden der Modelldateien: {e}")
        st.error("Stellen Sie sicher, dass folgende Dateien im 'models/' Ordner vorhanden sind:")
        st.error("- xgb_model_combined.joblib")
        st.error("- tfidf_vectorizer_combined.joblib") 
        st.error("- scaler_combined.joblib")
        st.error("- label_encoder_combined.joblib")
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

# ==== Einfachere Erklärungsfunktion ====
def analyze_prediction_simple(tweet_text, model, vectorizer, scaler, label_encoder, pred_encoded, X_all):
    """
    Einfachere und robustere Analyse der wichtigsten Einflussfaktoren
    """
    # SHAP Explainer erstellen
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    
    # Richtige SHAP-Werte extrahieren
    if isinstance(shap_values, list):
        if len(shap_values) > pred_encoded:
            shap_values_pred = shap_values[pred_encoded][0]
        else:
            shap_values_pred = shap_values[0][0]
    else:
        if len(shap_values.shape) == 2:
            shap_values_pred = shap_values[0]
        else:
            shap_values_pred = shap_values
    
    # Feature-Bereiche
    n_tfidf = len(vectorizer.get_feature_names_out())
    n_bert = 768
    n_eng = 14
    
    # Alle Features mit ihren Namen und SHAP-Werten
    all_features = []
    
    # 1. TF-IDF Features (Wörter)
    tfidf_features = vectorizer.get_feature_names_out()
    tfidf_values = X_all[0][:n_tfidf]
    tfidf_shap = shap_values_pred[:n_tfidf]
    
    for i, (word, tf_val, shap_val) in enumerate(zip(tfidf_features, tfidf_values, tfidf_shap)):
        if tf_val > 0:  # Nur Wörter die vorkommen
            all_features.append({
                'name': f"Wort: '{word}'",
                'type': 'word',
                'shap_value': shap_val,
                'feature_value': tf_val,
                'abs_shap': abs(shap_val),
                'word': word
            })
    
    # 2. BERT Features (als Gesamtblock)
    bert_shap = shap_values_pred[n_tfidf:n_tfidf+n_bert]
    bert_total = np.sum(bert_shap)
    all_features.append({
        'name': 'BERT Semantische Bedeutung',
        'type': 'bert',
        'shap_value': bert_total,
        'feature_value': 1.0,
        'abs_shap': abs(bert_total)
    })
    
    # 3. Engineered Features
    eng_shap = shap_values_pred[n_tfidf+n_bert:n_tfidf+n_bert+n_eng]
    eng_names = [
        "Tweet-Länge (Zeichen)", "Tweet-Länge (Wörter)", "Durchschnittliche Wortlänge", 
        "Großbuchstaben-Anteil", "Ausrufezeichen", "Fragezeichen", "Mehrfach-Satzzeichen", 
        "Politische Begriffe", "Emojis", "Hashtags", "Mentions", "URLs", "Punkte", "Ist Retweet"
    ]
    
    for name, shap_val in zip(eng_names, eng_shap):
        all_features.append({
            'name': name,
            'type': 'feature',
            'shap_value': shap_val,
            'feature_value': 1.0,
            'abs_shap': abs(shap_val)
        })
    
    return all_features, tweet_text

def display_simple_explanation(all_features, predicted_party, tweet_text, top_n=10):
    """
    Zeigt die wichtigsten Einflussfaktoren in einfacher Form
    """
    # Sortiere alle Features nach absolutem SHAP-Wert
    all_features.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    st.write("**🔍 Die wichtigsten Einflussfaktoren:**")
    
    # Debug Info
    if len(all_features) > 0:
        max_influence = max(f['abs_shap'] for f in all_features)
        st.write(f"_Debug: Stärkster Einfluss: {max_influence:.6f}_")
    
    # Zeige Top Features ohne Schwelle
    shown_count = 0
    words_shown = 0
    
    for i, feature in enumerate(all_features[:top_n*2]):  # Schaue mehr Features an
        if shown_count >= top_n:
            break
            
        # Begrenze Wörter auf max 5
        if feature['type'] == 'word' and words_shown >= 5:
            continue
            
        # Prüfe bei Wörtern ob sie im Text vorkommen
        if feature['type'] == 'word':
            if feature['word'].lower() not in tweet_text.lower():
                continue
        
        # Zeige Feature
        name = feature['name']
        shap_val = feature['shap_value']
        
        if shap_val > 0:
            st.write(f"🟢 **{name}** → unterstützt {predicted_party} (Einfluss: +{shap_val:.6f})")
        else:
            st.write(f"🔴 **{name}** → spricht gegen {predicted_party} (Einfluss: {shap_val:.6f})")
        
        if feature['type'] == 'word':
            words_shown += 1
        shown_count += 1
    
    # Falls immer noch nichts gezeigt wird
    if shown_count == 0:
        st.write("**🔍 Top 5 Faktoren (alle Werte):**")
        for i, feature in enumerate(all_features[:5]):
            name = feature['name']
            shap_val = feature['shap_value']
            st.write(f"{i+1}. **{name}**: {shap_val:.8f}")
    
    # Statistiken
    total_influence = sum(f['abs_shap'] for f in all_features)
    word_influence = sum(f['abs_shap'] for f in all_features if f['type'] == 'word')
    
    st.write("---")
    st.write(f"📊 **Einfluss-Verteilung:**")
    st.write(f"• Wörter: {word_influence:.6f}")
    st.write(f"• BERT Semantik: {next((f['abs_shap'] for f in all_features if f['type'] == 'bert'), 0):.6f}")
    st.write(f"• Tweet-Eigenschaften: {total_influence - word_influence:.6f}")
    st.write(f"• Gesamt: {total_influence:.6f}")



# ==== UI: Textfeld + Thema-Auswahl + Buttons ====
if "input_tweet" not in st.session_state:
    st.session_state["input_tweet"] = ""

thema = st.selectbox("📂 Wähle ein Thema:", list(SAMPLE_TWEET_CATEGORIES.keys()))

col1, col2 = st.columns([3, 1])
with col1:
    current_tweet_input = st.text_area(
        label="",
        placeholder="Gib einen Bundestags-Tweet ein...",
        height=100,
        label_visibility="collapsed",
        value=st.session_state["input_tweet"],
        key="input_tweet_widget"
    )
    st.session_state["input_tweet"] = current_tweet_input

with col2:
    if st.button("🔄 Beispiel-Tweet laden"):
        st.session_state["input_tweet"] = random.choice(SAMPLE_TWEET_CATEGORIES[thema])

predict_clicked = st.button("🔮 Vorhersagen")

# ==== Vorhersage und Erklärung ====
if predict_clicked and st.session_state["input_tweet"].strip():
    tweet_to_predict = st.session_state["input_tweet"]
    
    with st.spinner("Analysiere Tweet..."):
        # Features extrahieren
        # TF-IDF
        X_tfidf = vectorizer.transform([tweet_to_predict])
        # BERT
        X_bert = embed_single_text(tweet_to_predict)
        # Engineered Features
        X_eng = extract_features(tweet_to_predict)
        X_eng_scaled = scaler.transform(X_eng)
        # Kombiniere alle Features
        X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
        
        # Vorhersage
        pred_encoded = model.predict(X_all)[0]
        pred = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Wahrscheinlichkeiten
        probs = model.predict_proba(X_all)[0]
        
        # Erklärung generieren
        all_features, tweet_text = analyze_prediction_simple(
            tweet_to_predict, model, vectorizer, scaler, label_encoder, pred_encoded, X_all
        )
    
    # Ergebnisse anzeigen
    st.success(f"🟩 Vorhergesagte Partei: **{pred}**")
    
    # Informationen zur vorhergesagten Partei
    with st.expander(f"🧭 Informationen über {pred}"):
        st.write(PARTY_INFOS.get(pred, "Keine Informationen verfügbar."))
    
    # Wahrscheinlichkeitsverteilung
    st.subheader("📊 Vorhersagewahrscheinlichkeit")
    
    df_probs = pd.DataFrame({
        "Partei": label_encoder.inverse_transform(range(len(probs))),
        "Wahrscheinlichkeit": probs
    }).sort_values(by="Wahrscheinlichkeit", ascending=False)
    
    for index, row in df_probs.head(3).iterrows():
        color = PARTY_COLORS.get(row['Partei'], "#aaaaaa")
        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>{row['Partei']}</span>: {row['Wahrscheinlichkeit']:.2%}",
            unsafe_allow_html=True
        )
    
    # Einfache SHAP Erklärung
    st.subheader("🔍 Was hat die Entscheidung beeinflusst?")
    
    with st.spinner("Analysiere Einflüsse..."):
        # Zeige alle wichtigen Faktoren
        display_simple_explanation(all_features, pred, tweet_to_predict, top_n=10)
    
    # Tweet-Analyse Metriken
    st.subheader("📝 Tweet-Analyse")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Zeichen", len(tweet_to_predict))
        st.metric("Wörter", len(tweet_to_predict.split()))
        st.metric("Hashtags", count_hashtags(tweet_to_predict))
    
    with col2:
        st.metric("Mentions", count_mentions(tweet_to_predict))
        st.metric("URLs", count_urls(tweet_to_predict))
        st.metric("Emojis", count_emojis(tweet_to_predict))
    
    with col3:
        st.metric("Politische Begriffe", count_political_terms(tweet_to_predict))
        st.metric("Großbuchstaben-Anteil", f"{uppercase_ratio(tweet_to_predict):.1%}")
        retweet_status = "Ja" if is_retweet(tweet_to_predict) else "Nein"
        st.metric("Retweet", retweet_status)

st.markdown("---")
st.caption("📌 Dieses Tool wurde im Rahmen des ML4B-Projekts entwickelt – zur Parteivorhersage basierend auf deutsche Tweets.")