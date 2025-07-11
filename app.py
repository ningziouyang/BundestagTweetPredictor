import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import os # Stellen Sie sicher, dass das os-Modul importiert ist

# ==== Modellkonfiguration: Jetzt nur noch die Dateien, die Sie tatsächlich behalten haben ====
# app.py befindet sich im Verzeichnis streamlit_app/
# Die Modelldateien befinden sich im Verzeichnis streamlit_app/models/
# Daher ist BASE_DIR das Verzeichnis, in dem sich app.py befindet, und die Modelle sind unter BASE_DIR/models/
BASE_DIR = os.path.dirname(__file__)

MODEL_OPTIONS = {
    # Bitte wählen und kombinieren Sie die Modelloptionen entsprechend Ihren tatsächlichen Bedürfnissen
    # Ich werde die Optionen basierend auf den von Ihnen bereitgestellten Dateinamen neu erstellen
    # Hinweis: Dies sind spekulative Kombinationen basierend auf Ihren Modelldateien. Sie können sie an die tatsächliche Modelltraining und Verwendung anpassen

    # Option 1: Baseline TF-IDF Modell (angenommen unter Verwendung von lr_model_no_urls und tfidf_no_urls)
    "TF-IDF baseline (no_urls)": {
        "model": os.path.join(BASE_DIR, "models", "lr_model_no_urls.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_no_urls.joblib"), # Angenommen, dies ist der Vektorisierer für Ihre TF-IDF Baseline
        "scaler": None
    },
    # Option 2: TF-IDF + zusätzliche Merkmale (angenommen unter Verwendung von lr_model_extra_no_urls und scaler_extra_no_urls)
    "TF-IDF + Extra Features": {
        "model": os.path.join(BASE_DIR, "models", "lr_model_extra_no_urls.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_extra_no_urls.joblib"), # Angenommen, dies ist der Vektorisierer für Extra Features
        "scaler": os.path.join(BASE_DIR, "models", "scaler_extra_no_urls.joblib")
    },
    # Option 3: TF-IDF + BERT + Engineered (angenommen unter Verwendung von lr_tfidf_bert_engineered, tfidf_vectorizer_bert_engineered, feature_scaler_bert_engineered)
    "TF-IDF + BERT + Engineered": {
        "model": os.path.join(BASE_DIR, "models", "lr_tfidf_bert_engineered.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_vectorizer_bert_engineered.joblib"),
        "scaler": os.path.join(BASE_DIR, "models", "feature_scaler_bert_engineered.joblib")
    },
    # Option 4: Kombiniertes Modell (angenommen lr_model_combined, scaler_combined)
    "Combined Model": {
        "model": os.path.join(BASE_DIR, "models", "lr_model_combined.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_vectorizer_bert_engineered.joblib"), # Hier müssen Sie bestätigen, welchen Vektorisierer das kombinierte Modell verwendet
        "scaler": os.path.join(BASE_DIR, "models", "scaler_combined.joblib")
    }
}

st.title("Parteivorhersage für Bundestags-Tweets 🇩🇪")

choice = st.selectbox("📦 Wähle ein Modell:", list(MODEL_OPTIONS.keys()))
info = MODEL_OPTIONS[choice]

# **Wichtig: Fügen Sie vor dem Laden der Dateien eine Dateiexistenzprüfung hinzu**
try:
    # Überprüfen und Laden der Modelldatei
    model_path = info["model"]
    if not os.path.exists(model_path):
        st.error(f"❌ Fehler: Modelldatei '{os.path.basename(model_path)}' existiert nicht in '{os.path.dirname(model_path)}'. Bitte überprüfen Sie, ob die Datei hochgeladen wurde.")
        st.stop()
    model = joblib.load(model_path)

    # Überprüfen und Laden der Vektorisiererdatei
    vectorizer_path = info["vectorizer"]
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Fehler: Vektorisiererdatei '{os.path.basename(vectorizer_path)}' existiert nicht in '{os.path.dirname(vectorizer_path)}'. Bitte überprüfen Sie, ob die Datei hochgeladen wurde.")
        st.stop()
    vectorizer = joblib.load(vectorizer_path)

    # Überprüfen und Laden der Skaliererdatei
    scaler = None
    if info["scaler"]:
        scaler_path = info["scaler"]
        if not os.path.exists(scaler_path):
            st.error(f"❌ Fehler: Skaliererdatei '{os.path.basename(scaler_path)}' existiert nicht in '{os.path.dirname(scaler_path)}'. Bitte überprüfen Sie, ob die Datei hochgeladen wurde.")
            st.stop()
        scaler = joblib.load(scaler_path)

except FileNotFoundError as e:
    st.error(f"❌ Fehler: Modelldatei konnte nicht gefunden werden. Bitte überprüfen Sie den Dateipfad und ob die Modelldateien hochgeladen wurden.")
    st.error(f"Detaillierter Fehler: {e}")
    st.stop() # Beendet die Anwendungsausführung, um weitere Fehler zu vermeiden
except Exception as e: # Fängt andere mögliche Ladefehler ab, z.B. wenn die Datei beschädigt ist und joblib sie nicht parsen kann
    st.error(f"❌ Fehler: Beim Laden der Modelldatei ist ein Fehler aufgetreten. Die Datei könnte beschädigt oder unvollständig sein.")
    st.error(f"Detaillierter Fehler: {e}")
    st.stop()


use_bert = "BERT" in choice

# ==== BERT (Wenn das ausgewählte Modell BERT beinhaltet, laden) ====
if use_bert:
    try:
        # Hinweis: Das bert-base-german-cased Modell wird beim ersten Ausführen heruntergeladen.
        # Wenn die Bereitstellungsumgebung keinen Netzwerkzugriff hat oder der Download fehlschlägt, tritt hier ein Fehler auf.
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        bert_model = AutoModel.from_pretrained("bert-base-german-cased")
        bert_model.eval() # Setzt das Modell in den Evaluationsmodus
    except Exception as e:
        st.error(f"❌ BERT Modell oder dessen Tokenizer konnte nicht geladen werden. Bitte überprüfen Sie die Netzwerkverbindung oder den Modellnamen.")
        st.error(f"Detaillierter Fehler: {e}")
        st.stop()


# ==== Feature Engineering (Hilfsfunktionen für Feature Engineering) ====
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_emojis(text):
    try:
        # Versucht, die emoji-Bibliothek zu importieren; falls nicht installiert, greift es auf eine einfache Zählung zurück
        import emoji
        # Stellt sicher, dass der Text vom Typ String ist, um Fehler bei nicht-String-Eingaben zu vermeiden
        return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)
    except ImportError:
        # Wenn die emoji-Bibliothek nicht installiert ist, wird eine einfache Alternative verwendet
        return str(text).count(":")
    except TypeError: # Behandelt den Fall, dass der Text None oder ein anderer nicht iterierbarer Typ ist
        return 0

def extract_features(text):
    # Stellt sicher, dass der Text ein String ist, um Fehler im re-Modul zu vermeiden
    text_str = str(text)
    # Vermeidet Division durch Null
    words = re.findall(r"\w+", text_str)
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    uppercase_rat = sum(1 for c in text_str if c.isupper()) / len(text_str) if text_str else 0

    feats = [
        len(text_str),
        len(words),
        avg_word_len,
        uppercase_rat,
        text_str.count("!"),
        text_str.count("?"),
        len(re.findall(r"[!?]{2,}", text_str)),
        sum(1 for w in POLITICAL_TERMS if w in text_str.lower()),
        count_emojis(text_str),
        len(re.findall(r"#\w+", text_str)),
        len(re.findall(r"@\w+", text_str)),
        len(re.findall(r"http\S+|www\S+|https\S+", text_str)),
        len(re.findall(r"\.\.+", text_str)),
        int(text_str.strip().lower().startswith("rt @")),
    ]
    return np.array(feats).reshape(1, -1)

# Bert Embedding Funktion
def embed_single_text(text):
    with torch.no_grad(): # Beim Inferieren sind keine Gradientenberechnungen erforderlich
        # truncation=True schneidet zu lange Sequenzen ab, padding="max_length" füllt auf die maximale Länge auf
        # max_length=64 ist ein Beispielwert, den Sie an die Einstellungen Ihres BERT-Modells beim Training anpassen können
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        # Verschiebt die Eingabe auf dasselbe Gerät wie das Modell (z.B. GPU, falls verfügbar)
        # Wenn keine GPU vorhanden ist, wird dies standardmäßig auf der CPU ausgeführt
        output = bert_model(**encoded)
        # Ruft das Embedding des [CLS]-Tokens ab und konvertiert es in ein NumPy-Array
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)


# ==== UI-Eingabe und Vorhersagelogik ====
st.markdown("✏️ **Gib einen Bundestags-Tweet ein:**")
tweet = st.text_area("", placeholder="Wir fordern mehr Klimaschutz und soziale Gerechtigkeit für alle...")

if tweet and st.button("🔮 Vorhersagen"):
    # Sicherstellen, dass X_tfidf ein Array ist oder in ein Array umgewandelt werden kann, um mit np.hstack kompatibel zu sein
    X_tfidf = vectorizer.transform([tweet]).toarray() # Konvertiert in ein dichtes Array
    
    X_eng_scaled = None
    if scaler: # Nur wenn der Skalierer vorhanden ist, Merkmalsextraktion und Skalierung durchführen
        try:
            X_eng = extract_features(tweet)
            X_eng_scaled = scaler.transform(X_eng)
        except Exception as e:
            st.error(f"❌ Feature Engineering oder Skalierung fehlgeschlagen: {e}")
            st.stop()

    X_bert = None
    # Sicherstellen, dass use_bert True ist und bert_model erfolgreich geladen wurde
    if use_bert and 'bert_model' in locals():
        try:
            X_bert = embed_single_text(tweet)
        except Exception as e:
            st.error(f"❌ BERT Embedding fehlgeschlagen: {e}")
            st.stop()

    # === Merkmals-Kombination ===
    # Kombiniert die Eingabedaten für das Modell basierend auf dem ausgewählten Modell und den Merkmalen
    final_features = []
    final_features.append(X_tfidf)

    if X_bert is not None:
        final_features.append(X_bert)
    
    if X_eng_scaled is not None:
        final_features.append(X_eng_scaled)
    
    # Verwendet np.hstack, um alle Merkmals-Arrays zu kombinieren
    try:
        X_all = np.hstack(final_features)
    except ValueError as e:
        st.error(f"❌ Fehler beim Zusammenführen von Merkmalen. Bitte stellen Sie sicher, dass alle Merkmals-Arrays dimensionkompatibel sind: {e}")
        st.error(f"TF-IDF Shape: {X_tfidf.shape if 'X_tfidf' in locals() else 'N/A'}")
        st.error(f"BERT Shape: {X_bert.shape if 'X_bert' in locals() else 'N/A'}")
        st.error(f"Engineered Scaled Shape: {X_eng_scaled.shape if 'X_eng_scaled' in locals() else 'N/A'}")
        st.stop()


    # Vorhersage durchführen
    try:
        pred = model.predict(X_all)[0]
        st.success(f"🗳️ **Vorhergesagte Partei:** {pred}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_all)[0]
            st.subheader("📊 Wahrscheinlichkeit je Partei")
            # Verwenden Sie eine List-Comprehension, um sicherzustellen, dass Schlüssel und Werte native Python-Typen für die Streamlit-Plotting sind
            st.bar_chart({p: float(prob) for p, prob in zip(model.classes_, probs)})
    except Exception as e:
        st.error(f"❌ Fehler bei der Vorhersage: {e}")
        st.warning("Bitte stellen Sie sicher, dass alle Modelle und Skalierer-Dateien korrekt geladen und nicht leer sind und dass die Eingangsmerkmale die gleiche Dimension wie beim Modelltraining haben.")


st.markdown("---")
st.markdown("🔍 Dieses Tool kombiniert klassische Textmerkmale (TF-IDF), BERT-Embeddings und engineered Features zur Klassifikation von Bundestags-Tweets nach Partei.")