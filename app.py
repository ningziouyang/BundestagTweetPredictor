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

# ==== Modelloptionen und ihre Beschreibungen (Vorteile/Nachteile) ====
MODEL_OPTIONS = {
    "TF-IDF baseline": {
        "model": "models/lr_model_no_urls.joblib",
        "vectorizer": "models/tfidf_no_urls.joblib",
        "scaler": None,
        "description": """
        **Vorteile:**
        * **Einfach und schnell:** Training und Vorhersage sind schnell, benötigen wenig Rechenressourcen.
        * **Leicht verständlich:** Basierend auf Worthäufigkeit und inverser Dokumentenfrequenz, spiegelt direkt die Wortbedeutung wider.
        * **Basismodell:** Gut geeignet als Leistungsreferenz für komplexere Modelle.
        
        **Nachteile:**
        * **Ignoriert Wortreihenfolge und Kontext:** Versteht keine semantischen Beziehungen zwischen Wörtern oder Satzstrukturen.
        * **Dimensionalitätsfluch:** Bei großem Wortschatz sind die Feature-Vektoren hochdimensional, was zu Sparsity-Problemen führen kann.
        * **Fehlende semantische Informationen:** Kann fortgeschrittene semantische Informationen wie Synonyme oder Antonyme nicht erfassen.
        """
    },
    "TF-IDF + Extra Features": {
        "model": "models/lr_model_extra_no_urls.joblib",
        "vectorizer": "models/tfidf_extra_no_urls.joblib",
        "scaler": "models/scaler_extra_no_urls.joblib",
        "description": """
        **Vorteile:**
        * **Verbesserte Ausdrucksfähigkeit:** Einführung zusätzlicher Features (z.B. Anzahl Emojis, Hashtags, URLs), die von TF-IDF allein nicht erfasst werden.
        * **Erfassung des Textstils:** Diese zusätzlichen Features können „nicht-inhaltliche“ Informationen über den Tweet widerspiegeln, wie die Neigung oder Aktivität des Verfassers.
        * **Relativ effizient:** Geringere Rechenkosten als reine BERT-Modelle, während die Leistung bis zu einem gewissen Grad verbessert wird.
        
        **Nachteile:**
        * **Feature Engineering ist erfahrungsabhängig:** Die Auswahl und Gestaltung der zusätzlichen Features erfordert Fachwissen und experimentelle Validierung.
        * **Ignoriert weiterhin tiefe Semantik:** Der Kern bleibt TF-IDF, das Verständnis komplexer semantischer Beziehungen zwischen Wörtern ist weiterhin begrenzt.
        * **Sparsity-Probleme könnten weiterhin bestehen:** Wenn die zusätzlichen Features nicht eng mit dem Textinhalt verbunden sind, ist die Leistungssteigerung möglicherweise nicht signifikant.
        """
    },
    "TF-IDF + BERT + Engineered": {
        "model": "models/lr_tfidf_bert_engineered.joblib",
        "vectorizer": "models/tfidf_vectorizer_bert_engineered.joblib",
        "scaler": "models/feature_scaler_bert_engineered.joblib",
        "description": """
        **Vorteile:**
        * **Kombination vielfältiger Informationen:** Vereint die Wichtigkeit von Wörtern (TF-IDF), das tiefe semantische Verständnis (BERT) und den Textstil (handgefertigte Features).
        * **Starkes semantisches Verständnis:** Das BERT-Modell kann die kontextuelle Bedeutung von Wörtern und komplexe Satzbeziehungen erfassen, was das Verständnis des Textinhalts erheblich verbessert.
        * **Meist beste Leistung:** Bei den meisten Textklassifizierungsaufgaben erzielt dieses kombinierte Modell eine höhere Genauigkeit und Robustheit.
        
        **Nachteile:**
        * **Hoher Rechenressourcenverbrauch:** BERT-Modelle sind groß und erfordern mehr Speicher und Rechenleistung (insbesondere GPUs).
        * **Langsames Training und Vorhersage:** Im Vergleich zu reinen TF-IDF-Modellen steigt die Trainings- und Vorhersagezeit erheblich.
        * **Hohe Modellkomplexität:** Besteht aus mehreren Komponenten, was das Debuggen und Optimieren relativ komplex macht.
        """
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

# ==== Layout-Konfiguration ====
st.set_page_config(page_title="Parteivorhersage", layout="wide")

# Korrektur für den oberen Leerraum:
# Überprüfe, ob hier leere st.write(), st.markdown("") oder ähnliches eingefügt wurde.
# Wenn ja, entferne sie. Standardmäßig hat Streamlit einen kleinen oberen Abstand.
# Manchmal hilft es, CSS für das main-Container anzupassen, aber das ist selten nötig.
# Beispiel für potenziellen Fehler, der hier entfernt werden sollte:
# st.write("")

st.markdown("""
    <style>
    /* CSS, um den Button neben dem Textbereich auszublenden */
    textarea + div[role='button'] {
        display: none !important;
    }
    /* Schriftgröße des Textbereichs anpassen */
    .element-container textarea {
        font-size: 16px !important;
    }
    /* Label des Textbereichs ausblenden */
    div[data-testid=stTextArea] label {
        display: none;
    }
    /* Um den vom Streamlit erzeugten oberen Leerraum zu reduzieren */
    /* Sei vorsichtig mit globalen CSS-Änderungen, sie können andere Elemente beeinflussen */
    .css-18e3th9 { /* Dies ist der Selektor für den main-Container in Streamlit */
        padding-top: 1rem; /* Reduziert den oberen Abstand. Standard ist oft 3rem oder mehr */
    }
    </style>
""", unsafe_allow_html=True)
st.title("🗳️ Parteivorhersage für Bundestags-Tweets")

# ==== Seitenleiste: Parteiinformationen ====
st.sidebar.header("ℹ️ Parteiinformationen")
for partei, info in PARTY_INFOS.items():
    with st.sidebar.expander(partei):
        st.write(info)

# ==== Modellauswahl & Laden der Modelle ====
choice = st.selectbox("🔍 Wähle ein Modell:", list(MODEL_OPTIONS.keys()))
info = MODEL_OPTIONS[choice]

# Anzeige der Modellvor- und -nachteile in einem Expander
with st.expander(f"✨ **{choice} - Modellmerkmale**"):
    st.markdown(info["description"])

# Laden der Modelle und Komponenten
# Es ist wichtig, dass die Pfade zu deinen Modellen korrekt sind,
# z.B. models/lr_model_no_urls.joblib
try:
    model = joblib.load(info["model"])
    vectorizer = joblib.load(info["vectorizer"])
    scaler = joblib.load(info["scaler"]) if info["scaler"] else None
    use_bert = "BERT" in choice
except FileNotFoundError as e:
    st.error(f"Fehler beim Laden der Modelldatei: {e}. Bitte stellen Sie sicher, dass die Dateien im 'models/' Ordner vorhanden sind.")
    st.stop() # Stoppt die Ausführung der App, wenn Modelle nicht gefunden werden

if use_bert:
    # BERT-Modell und Tokenizer nur bei Bedarf laden und cachen
    @st.cache_resource
    def load_bert_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
            bert_model = AutoModel.from_pretrained("bert-base-german-cased")
            bert_model.eval() # Setzt das Modell in den Evaluationsmodus
            return tokenizer, bert_model
        except Exception as e:
            st.error(f"Fehler beim Laden des BERT-Modells: {e}. Stellen Sie sicher, dass Sie eine Internetverbindung haben oder das Modell lokal verfügbar ist.")
            st.stop() # Stoppt die App, wenn BERT nicht geladen werden kann
    tokenizer, bert_model = load_bert_model()

# ==== Feature-Extraktion ====
# Liste politischer Begriffe für Feature Engineering
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

# Funktionen zur Extraktion verschiedener Textmerkmale
def count_emojis(text): return str(text).count("😀")
def avg_word_length(text):
    words = re.findall(r"\w+", str(text))
    return sum(len(w) for w in words) / len(words) if words else 0 if len(words) > 0 else 0 # Defensive Programmierung
def uppercase_ratio(text): return sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
def multi_punct_count(text): return len(re.findall(r"[!?]{2,}", str(text)))
def count_political_terms(text): return sum(1 for w in POLITICAL_TERMS if w in str(text).lower())
def count_hashtags(text): return len(re.findall(r"#\w+", str(text)))
def count_mentions(text): return len(re.findall(r"@\w+", str(text)))
def count_urls(text): return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))
def count_dots(text): return len(re.findall(r"\.\.+", str(text)))
def is_retweet(text): return int(str(text).strip().lower().startswith("rt @"))

# Funktion zur Extraktion der vollständigen Feature-Liste
def extract_features(text):
    return np.array([[len(str(text)), len(str(text).split()), avg_word_length(text), uppercase_ratio(text),
                     str(text).count("!"), str(text).count("?"), multi_punct_count(text), count_political_terms(text),
                     count_emojis(text), count_hashtags(text), count_mentions(text), count_urls(text),
                     count_dots(text), is_retweet(text)]])

# Funktion zur Extraktion der "Extra Features"
def extract_extra_features(text):
    return np.array([[count_emojis(text), count_hashtags(text), count_mentions(text), count_urls(text)]])

# Funktion zur Generierung von BERT-Embeddings
def embed_single_text(text):
    with torch.no_grad(): # Deaktiviert die Gradientenberechnung für die Inferenz
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        output = bert_model(**encoded)
        # Nimmt das Embedding des [CLS]-Tokens (erstes Token) als Satz-Embedding
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

# ==== UI: Textfeld + Thema-Auswahl + Buttons ====
# Sicherstellen, dass 'input_tweet' im session_state initialisiert ist
if "input_tweet" not in st.session_state:
    st.session_state["input_tweet"] = "" # Initialwert ist ein leerer String

thema = st.selectbox("📂 Wähle ein Thema:", list(SAMPLE_TWEET_CATEGORIES.keys()))

col1, col2 = st.columns([3, 1])
with col1:
    # Textbereich für die Eingabe des Tweets
    current_tweet_input = st.text_area(
        label="", # Label ist ausgeblendet
        placeholder="Gib einen Bundestags-Tweet ein...",
        height=100,
        label_visibility="collapsed",
        value=st.session_state["input_tweet"], # Der Wert wird aus dem session_state gelesen
        key="input_tweet_widget" # Eindeutiger Key für das Widget
    )
    # Aktualisiert den session_state, wenn der Benutzer im Textfeld tippt
    st.session_state["input_tweet"] = current_tweet_input

with col2:
    # Button zum Laden eines Beispiel-Tweets
    if st.button("🔄 Beispiel-Tweet laden"):
        # Aktualisiert den session_state mit einem zufälligen Beispiel-Tweet
        st.session_state["input_tweet"] = random.choice(SAMPLE_TWEET_CATEGORIES[thema])
        # Streamlit wird das Skript automatisch neu ausführen, um die Änderung zu reflektieren

# Button zur Vorhersage
predict_clicked = st.button("🔮 Vorhersagen")

# Logik für die Vorhersage, wenn der Button geklickt wurde und der Tweet nicht leer ist
if predict_clicked and st.session_state["input_tweet"].strip(): # Nutzt den Tweet aus dem session_state
    tweet_to_predict = st.session_state["input_tweet"]

    # TF-IDF Features transformieren
    X_tfidf = vectorizer.transform([tweet_to_predict])

    X_eng_scaled = None
    # Skalierte Ingenieur-Features extrahieren, falls ein Scaler vorhanden ist
    if scaler:
        # Je nach Modellwahl werden unterschiedliche Ingenieur-Features extrahiert
        X_eng = extract_extra_features(tweet_to_predict) if "Extra Features" in choice else extract_features(tweet_to_predict)
        X_eng_scaled = scaler.transform(X_eng)

    # Zusammenführen aller Features basierend auf der Modellauswahl
    if use_bert:
        X_bert = embed_single_text(tweet_to_predict)
        # Stapelt TF-IDF, BERT und skalierte Ingenieur-Features horizontal
        X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
    elif scaler:
        # Stapelt TF-IDF und skalierte Ingenieur-Features horizontal
        X_all = np.hstack([X_tfidf.toarray(), X_eng_scaled])
    else:
        # Nur TF-IDF Features
        X_all = X_tfidf

    # Modellvorhersage durchführen
    pred = model.predict(X_all)[0]
    st.success(f"🟩 Vorhergesagte Partei: **{pred}**")

    # Informationen zur vorhergesagten Partei anzeigen
    with st.expander(f"🧭 Informationen über {pred}"):
        st.write(PARTY_INFOS.get(pred, "Keine Informationen verfügbar."))

    # Wahrscheinlichkeitsverteilung anzeigen, falls das Modell predict_proba unterstützt
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_all)[0]
        # DataFrame für die Visualisierung der Wahrscheinlichkeiten erstellen
        df = pd.DataFrame({
            "Partei": model.classes_,
            "Wahrscheinlichkeit": probs,
            "Farbe": [PARTY_COLORS.get(p, "#aaaaaa") for p in model.classes_]
        })
        # Nach Wahrscheinlichkeit absteigend sortieren
        df = df.sort_values(by="Wahrscheinlichkeit", ascending=False)

        st.subheader("📊 Vorhersagewahrscheinlichkeit (Top 3)")

        # Die Top 3 Parteien anzeigen und formatieren
        top_3_parties = df.head(3)
        for index, row in top_3_parties.iterrows():
            st.markdown(
                f"<span style='color:{row['Farbe']}; font-weight:bold;'>{row['Partei']}</span>: {row['Wahrscheinlichkeit']:.2%}",
                unsafe_allow_html=True
            )
        
        # Optional: Den vollständigen Balkenchart weiterhin anzeigen, wenn gewünscht
        with st.expander("Vollständige Wahrscheinlichkeitsverteilung anzeigen"):
            st.bar_chart(data=df.set_index("Partei")["Wahrscheinlichkeit"])

st.markdown("---")
st.caption("📌 Dieses Tool wurde im Rahmen des ML4B-Projekts entwickelt – zur Parteivorhersage deutscher Bundestags-Tweets.")
