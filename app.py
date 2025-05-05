import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 💾 Load model
model = load_model('best_model.h5')

# ✨ Class labels — match your training folder names
class_names = ['Grenade', 'Gun', 'Knife']

# 🎨 App layout
st.set_page_config(page_title="Weapon Classifier", layout="centered")
st.title("🔍 Weapon Image Classifier")
st.markdown("Upload an image of a **knife**, **handgun**, or **grenade** and let the model predict what it is.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # 🧼 Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🤖 Predict
    prediction = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # 📊 Display result
    st.markdown(f"### ✅ Prediction: `{pred_class.upper()}`")
    st.progress(int(confidence * 100))

    # Optional: show probabilities
    st.markdown("#### Confidence Scores:")
    for i, score in enumerate(prediction):
        st.write(f"- {class_names[i]}: **{score*100:.2f}%**")
