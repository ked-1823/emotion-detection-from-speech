# Speech Emotion Detection

This project focuses on detecting human emotions directly from speech using deep learning.  
Instead of relying only on text sentiment analysis, this system learns to recognize emotions such as **happy, sad, angry, neutral, fear, disgust** from raw audio signals.  

---

## 🔹 Project Overview
The model takes raw audio files as input, extracts features, and predicts the underlying emotion.  
It is trained on datasets like [Toronto Emotional Speech Set (TESS)] or other publicly available speech emotion datasets.

---

## 🔹 Features
- Converts audio into **spectrograms** and **MFCCs (Mel-frequency cepstral coefficients)**  
- Trains a **Convolutional Neural Network (CNN)** on spectrogram images  
- Handles preprocessing (normalization, trimming, noise handling)  
- Provides predictions on unseen real-world audio samples  

---

## 🔹 Tech Stack
- **Python**  
- **TensorFlow / Keras**  
- **Librosa** for audio processing  
- **NumPy, Pandas, Matplotlib** for data handling and visualization  

---

## 🔹 Installation

Clone the repository:
```bash
git clone https://github.com/your-username/speech-emotion-detection.git
cd emotion-detection-from-speech
