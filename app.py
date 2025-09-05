import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# -------------------------
# 1) Load your trained model
# -------------------------
model = tf.keras.models.load_model("best_model.h5")

# Replace this with your encoder categories
enc_categories = ['neutral','happy','sad','angry','fear']  # Example

# -------------------------
# 2) Normalize audio
# -------------------------
def normalize_audio(y):
    return y / np.max(np.abs(y))

# -------------------------
# 3) Extract MFCC
# -------------------------
def extract_mfcc_real(y, sr, max_len=130, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc

# -------------------------
# 4) Predict emotion
# -------------------------
def predict_emotion_audio(audio, sr):
    audio = normalize_audio(audio)
    mfcc = extract_mfcc_real(audio, sr)
    x_input = np.expand_dims(mfcc, axis=0)
    pred = model.predict(x_input, verbose=0)
    pred_class = np.argmax(pred[0])
    return enc_categories[pred_class]

# -------------------------
# 5) Streamlit UI
# -------------------------
st.title("🎤 Voice Emotion Detection")

st.write("Upload a short audio file (around 3 seconds) for emotion prediction.")

uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)
    predicted_emotion = predict_emotion_audio(y, sr)
    st.write(f"Predicted Emotion: **{predicted_emotion}**")
