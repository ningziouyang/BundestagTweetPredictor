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

st.title("🗳️ Parteivorhersage für Bundestags-Tweets")

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

# ==== Verbesserte SHAP Erklärungsfunktion ====
def explain_prediction_improved(tweet_text, model, vectorizer, scaler, label_encoder, pred_encoded, X_all):
    """
    Verbesserte Erklärung der Vorhersage mit fokussierter Wortanalyse
    """
    # SHAP Explainer erstellen
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    
    # Richtige SHAP-Werte für die vorhergesagte Klasse extrahieren
    if isinstance(shap_values, list):
        if len(shap_values) > pred_encoded:
            shap_values_for_prediction = shap_values[pred_encoded][0]
        else:
            shap_values_for_prediction = shap_values[0][0]
    else:
        if len(shap_values.shape) == 2:
            shap_values_for_prediction = shap_values[0]
        else:
            shap_values_for_prediction = shap_values
    
    # Feature-Bereiche definieren
    n_tfidf = len(vectorizer.get_feature_names_out())
    n_bert = 768
    n_eng = 14
    
    # TF-IDF Features und ihre Werte
    tfidf_shap = shap_values_for_prediction[:n_tfidf]
    tfidf_features = vectorizer.get_feature_names_out()
    tfidf_values = X_all[0][:n_tfidf]
    
    # BERT Features
    bert_shap = shap_values_for_prediction[n_tfidf:n_tfidf+n_bert]
    bert_importance = np.sum(np.abs(bert_shap))
    
    # Engineered Features
    eng_shap = shap_values_for_prediction[n_tfidf+n_bert:n_tfidf+n_bert+n_eng]
    eng_feature_names = [
        "Zeichen-Anzahl", "Wort-Anzahl", "Durchschnittliche Wortlänge", "Großbuchstaben-Anteil",
        "Ausrufezeichen", "Fragezeichen", "Mehrfach-Satzzeichen", "Politische Begriffe", 
        "Emojis", "Hashtags", "Mentions", "URLs", "Punkte", "Ist Retweet"
    ]
    
    return {
        'tfidf_shap': tfidf_shap,
        'tfidf_features': tfidf_features,
        'tfidf_values': tfidf_values,
        'bert_importance': bert_importance,
        'eng_shap': eng_shap,
        'eng_feature_names': eng_feature_names,
        'original_text': tweet_text
    }

def display_word_importance(explanation, predicted_party, top_n=10):
    """
    Zeigt die wichtigsten Wörter für die Vorhersage an
    """
    # Finde Wörter die im Tweet vorkommen (TF-IDF > 0)
    word_impacts = []
    for word, shap_val, tf_val in zip(explanation['tfidf_features'], 
                                      explanation['tfidf_shap'], 
                                      explanation['tfidf_values']):
        if tf_val > 0:  # Wort kommt vor
            word_impacts.append({
                'word': word,
                'shap_value': shap_val,
                'tf_value': tf_val,
                'abs_shap': abs(shap_val)
            })
    
    # Sortiere nach absolutem SHAP-Wert
    word_impacts.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    st.write("**📝 Einflussreiche Wörter:**")
    
    if not word_impacts:
        st.write("_Keine spezifischen Wörter mit messbarem Einfluss gefunden._")
        return
    
    # Berechne dynamische Schwelle basierend auf der Verteilung
    shap_values = [w['abs_shap'] for w in word_impacts]
    if len(shap_values) > 0:
        threshold = np.percentile(shap_values, 70)  # Top 30%
        threshold = max(threshold, 0.001)  # Minimalschwelle
    else:
        threshold = 0.001
    
    # Zeige wichtige Wörter
    shown_words = 0
    for word_data in word_impacts[:top_n]:
        if word_data['abs_shap'] > threshold and shown_words < top_n:
            word = word_data['word']
            shap_val = word_data['shap_value']
            
            # Prüfe ob das Wort tatsächlich im originalen Text vorkommt
            if word.lower() in explanation['original_text'].lower():
                if shap_val > 0:
                    st.write(f"🟢 **'{word}'** → unterstützt {predicted_party} (Einfluss: +{shap_val:.4f})")
                else:
                    st.write(f"🔴 **'{word}'** → spricht gegen {predicted_party} (Einfluss: {shap_val:.4f})")
                shown_words += 1
    
    # Falls keine Wörter über der Schwelle, zeige die Top 5
    if shown_words == 0:
        st.write("_Zeige die 5 einflussreichsten Wörter:_")
        for word_data in word_impacts[:5]:
            word = word_data['word']
            shap_val = word_data['shap_value']
            
            if word.lower() in explanation['original_text'].lower():
                if shap_val > 0:
                    st.write(f"🟢 **'{word}'** → +{shap_val:.4f}")
                else:
                    st.write(f"🔴 **'{word}'** → {shap_val:.4f}")
    
    # Zusätzliche Statistiken
    total_word_influence = sum(w['abs_shap'] for w in word_impacts)
    st.write(f"_Gesamteinfluss aller Wörter: {total_word_influence:.4f}_")
    
    return word_impacts

def display_feature_importance(explanation, predicted_party, top_n=5):
    """
    Zeigt die wichtigsten Tweet-Eigenschaften
    """
    st.write("**🔧 Wichtige Tweet-Eigenschaften:**")
    
    # Erstelle Feature-Impact Liste
    feature_impacts = []
    for name, shap_val in zip(explanation['eng_feature_names'], explanation['eng_shap']):
        feature_impacts.append({
            'name': name,
            'shap_value': shap_val,
            'abs_shap': abs(shap_val)
        })
    
    # Sortiere nach absolutem Einfluss
    feature_impacts.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    # Zeige Top Features
    for feature_data in feature_impacts[:top_n]:
        if feature_data['abs_shap'] > 0.001:  # Schwelle für Relevanz
            name = feature_data['name']
            shap_val = feature_data['shap_value']
            
            if shap_val > 0:
                st.write(f"🟢 **{name}** → unterstützt {predicted_party} (Einfluss: +{shap_val:.4f})")
            else:
                st.write(f"🔴 **{name}** → spricht gegen {predicted_party} (Einfluss: {shap_val:.4f})")
    
    # BERT Einfluss
    if explanation['bert_importance'] > 0.01:
        st.write(f"🧠 **BERT Semantik** → Gesamteinfluss: {explanation['bert_importance']:.4f}")

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
        explanation = explain_prediction_improved(
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
    
    # Verbesserte SHAP Erklärung
    st.subheader("🔍 Was hat die Entscheidung beeinflusst?")
    
    with st.spinner("Analysiere Einflüsse..."):
        # Zeige Wort-Einflüsse
        word_impacts = display_word_importance(explanation, pred, top_n=10)
        
        st.write("---")
        
        # Zeige Feature-Einflüsse
        display_feature_importance(explanation, pred, top_n=5)
    
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
st.caption("📌 Dieses Tool wurde im Rahmen des ML4B-Projekts entwickelt – zur Parteivorhersage deutscher Bundestags-Tweets.")