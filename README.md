# emotion4class-va

This repository contains a **real-time emotion recognition model** that uses **EEG and ECG signals** to classify emotions into **4 valence-arousal categories**:

- High Valence High Arousal (HVHA)
- High Valence Low Arousal (HVLA)
- - Low Valence Low Arousal (LVLV)
- Low Valence High Arousal (LVHA)

### 🧠 Key Features

- Supports **real-time inference** using physiological signals (EEG + ECG)
- Adaptively learns from new subjects during deployment to address **inter-subject variability**
- Lightweight and optimized for deployment using **Docker**
- Optionally integrates with **LabStreamingLayer (LSL)** for live data streaming

---


## 🧠 Emotion Classes

| Class | Description             |
|-------|-------------------------|
| HVHA  | Excited, Happy          |            |
| HVLA  | Calm, Peaceful          |
| LVLV  | Tired, Sad  
| LVHA  | Angry, Stressed         |
