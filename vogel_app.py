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

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.block-container {
    max-width: 750px;
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 20px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f8f9fa;
    margin-top: 20px;
}

.success {
    border-left: 6px solid #34a853;
}

.warning {
    border-left: 6px solid #fbbc05;
}

.error {
    border-left: 6px solid #ea4335;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown("<div class='title'>🐦 Vogel-KI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Erkennt Vögel mit deiner eigenen KI</div>", unsafe_allow_html=True)

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
        if " " in line and line.split(" ")[0].isdigit():
            labels.append(" ".join(line.split(" ")[1:]))
        else:
            labels.append(line)

    return labels

labels = load_labels()

# -------------------------
# PREPROCESS
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
bird_classes = [l for l in labels if "kein" not in l.lower()]

# -------------------------
# UPLOAD
# -------------------------
st.subheader("📤 Bild hochladen")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Dein Bild", use_column_width=True)

    st.divider()

    with st.spinner("🧠 KI analysiert..."):
        processed = preprocess(image)
        prediction = model.predict(processed)[0]

    # Top Prediction
    top_index = int(np.argmax(prediction))
    best_label = labels[top_index]
    best_conf = float(prediction[top_index])

    # Entscheidung
    is_bird = (best_label in bird_classes) and (best_conf >= THRESHOLD)

    # -------------------------
    # RESULT CARD
    # -------------------------
    st.subheader("📊 Ergebnis")

    if is_bird:
        st.markdown(f"""
        <div class='card success'>
            🐦 <b>Vogel erkannt</b><br>
            {best_label}
        </div>
        """, unsafe_allow_html=True)
    elif best_label in bird_classes:
        st.markdown(f"""
        <div class='card warning'>
            ⚠️ <b>Unsicher</b><br>
            {best_label}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='card error'>
            ❌ <b>Kein Vogel</b><br>
            {best_label}
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # CONFIDENCE VISUAL
    # -------------------------
    st.write("### 🔎 Sicherheit")
    st.progress(int(best_conf * 100))
    st.write(f"{round(best_conf * 100, 2)}%")

    # -------------------------
    # TOP 3 VISUAL
    # -------------------------
    st.write("### 📊 Top Vorhersagen")

    top_indices = prediction.argsort()[-3:][::-1]

    for i in top_indices:
        label = labels[i]
        conf = float(prediction[i])

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(label)
            st.progress(int(conf * 100))
        with col2:
            st.write(f"{round(conf*100,1)}%")

    st.divider()

    # -------------------------
    # DETAILS
    # -------------------------
    with st.expander("🔍 Alle Klassen anzeigen"):
        for i, score in enumerate(prediction):
            st.write(f"{labels[i]} – {round(float(score)*100,2)}%")
