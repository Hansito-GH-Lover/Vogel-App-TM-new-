import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Vogel-KI (Custom)",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 Deine eigene Vogel-KI")
st.write("Erkennt Vögel basierend auf deinem trainierten Modell.")

# -------------------------
# MODEL LADEN
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

try:
    model = load_model()
except Exception as e:
    st.error("❌ Modell konnte nicht geladen werden.")
    st.exception(e)
    st.stop()

# -------------------------
# LABELS LADEN (robust)
# -------------------------
@st.cache_resource
def load_labels():
    with open("labels.txt", "r") as f:
        raw = f.readlines()

    labels = []
    for line in raw:
        line = line.strip()

        # Fall 1: "0 Vogel"
        if " " in line and line.split(" ")[0].isdigit():
            labels.append(" ".join(line.split(" ")[1:]))

        # Fall 2: "Vogel"
        else:
            labels.append(line)

    return labels

labels = load_labels()

# -------------------------
# PREPROCESSING (TM STANDARD!)
# -------------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)

    # WICHTIG für Teachable Machine
    img_array = (img_array / 127.5) - 1

    return np.expand_dims(img_array, axis=0)

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Dein Bild", use_column_width=True)

        with st.spinner("🧠 Analyse läuft..."):
            processed = preprocess(image)
            prediction = model.predict(processed)[0]

        # Beste Klasse
        top_index = int(np.argmax(prediction))
        best_label = labels[top_index]
        best_conf = float(prediction[top_index])

        # -------------------------
        # OUTPUT
        # -------------------------
        st.subheader("📊 Ergebnis")

        st.success(f"🎯 {best_label}")
        st.write(f"🔎 Sicherheit: {round(best_conf * 100, 2)}%")
        st.progress(int(best_conf * 100))

        # -------------------------
        # ALLE KLASSEN
        # -------------------------
        with st.expander("🔍 Alle Vorhersagen"):
            for i, score in enumerate(prediction):
                st.write(f"{labels[i]} – {round(float(score) * 100, 2)}%")

    except Exception as e:
        st.error("❌ Fehler bei der Bildverarbeitung.")
        st.exception(e)
