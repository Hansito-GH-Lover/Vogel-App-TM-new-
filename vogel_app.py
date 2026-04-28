import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Vogel-KI",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 Vogel-KI")
st.write("Erkennt Vögel mit deinem eigenen Modell")

st.divider()

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

# -------------------------
# LABELS
# -------------------------
@st.cache_resource
def load_labels():
    with open("labels.txt", "r") as f:
        raw = f.readlines()

    labels = []
    for line in raw:
        line = line.strip()

        # "0 Vogel" → "Vogel"
        if " " in line and line.split(" ")[0].isdigit():
            labels.append(" ".join(line.split(" ")[1:]))
        else:
            labels.append(line)

    return labels

labels = load_labels()

# -------------------------
# PREPROCESSING (TM)
# -------------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 127.5) - 1
    return np.expand_dims(img_array, axis=0)

# -------------------------
# SETTINGS
# -------------------------
THRESHOLD = 0.75

# Alle Klassen außer "kein vogel" gelten als Vogel
bird_classes = [l for l in labels if "kein" not in l.lower()]

# -------------------------
# UPLOAD
# -------------------------
st.subheader("📤 Bild hochladen")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Dein Bild", use_column_width=True)

    st.divider()

    with st.spinner("🧠 KI analysiert..."):
        processed = preprocess(image)
        prediction = model.predict(processed)[0]

    # -------------------------
    # BESTE KLASSE
    # -------------------------
    top_index = int(np.argmax(prediction))
    best_label = labels[top_index]
    best_conf = float(prediction[top_index])

    # -------------------------
    # ENTSCHEIDUNG
    # -------------------------
    is_bird = (best_label in bird_classes) and (best_conf >= THRESHOLD)

    st.subheader("📊 Ergebnis")

    if is_bird:
        st.success(f"🐦 Vogel erkannt: {best_label}")
    elif best_label in bird_classes:
        st.warning(f"⚠️ Unsicher: {best_label}")
    else:
        st.error(f"❌ Kein Vogel erkannt")

    st.write(f"**Sicherheit:** {round(best_conf * 100, 2)}%")
    st.progress(int(best_conf * 100))

    st.divider()

    # -------------------------
    # DETAILS
    # -------------------------
    with st.expander("🔍 Alle Vorhersagen"):
        for i, score in enumerate(prediction):
            st.write(f"{labels[i]} – {round(float(score) * 100, 2)}%")
