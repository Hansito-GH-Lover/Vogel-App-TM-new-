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
# STYLE (clean & modern)
# -------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 700px;
}

h1 {
    text-align: center;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 18px;
}

.success-box {
    background-color: #e6f4ea;
    border-left: 6px solid #34a853;
}

.warning-box {
    background-color: #fdecea;
    border-left: 6px solid #ea4335;
}

.subtle {
    color: #666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("🐦 Vogel-KI")
st.markdown("<p class='subtle'>Erkennt Vögel mit deiner eigenen KI</p>", unsafe_allow_html=True)

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

    top_index = int(np.argmax(prediction))
    label = labels[top_index]
    confidence = float(prediction[top_index])

    # -------------------------
    # RESULT
    # -------------------------
    st.subheader("📊 Ergebnis")

    if "vogel" in label.lower():
        st.markdown(f"""
        <div class="result-box success-box">
            🐦 <b>Vogel erkannt</b><br>
            {label}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box warning-box">
            ❌ <b>Kein Vogel erkannt</b><br>
            {label}
        </div>
        """, unsafe_allow_html=True)

    st.write(f"**Sicherheit:** {round(confidence*100,2)}%")
    st.progress(int(confidence * 100))

    st.divider()

    # -------------------------
    # DETAILS
    # -------------------------
    with st.expander("🔍 Alle Vorhersagen anzeigen"):
        for i, score in enumerate(prediction):
            st.write(f"{labels[i]} – {round(float(score)*100,2)}%")
