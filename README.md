# Speech Emotion Recognition (SER)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Librosa](https://img.shields.io/badge/Librosa-0.10+-red.svg)

## Project Overview
This project implements a **Speech Emotion Recognition (SER)** system using digital signal processing and deep learning. The model identifies human emotions from audio recordings by analyzing spectral features.

## Dataset
The system is built to work with the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. It classifies speech into 8 distinct emotional categories:
- Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

## Key Features
- **Signal Processing**: Uses `librosa` to extract **MFCCs (Mel-Frequency Cepstral Coefficients)**, capturing the unique timbre and tonality of human speech.
- **Deep Learning Architecture**: Utilizes a **1D Convolutional Neural Network (CNN)** designed for sequence analysis.
- **Performance Visualization**: Automatically generates accuracy and loss plots to monitor model convergence.

## Directory Structure
```text
.
├── src/
│   ├── data_prep.py     # MFCC feature extraction logic
│   └── model.py         # 1D CNN architecture definition
├── data/
│   └── ravdess/         # RAVDESS dataset audio files (.wav)
├── outputs/             # Model performance plots
├── main.py              # Main orchestration script
└── requirements.txt     # Python dependencies
```

## How to Run

### 1. Requirements
Ensure you have Python installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your RAVDESS `.wav` files are placed in the `data/ravdess/` directory.

### 3. Execute
Run the end-to-end pipeline:
```bash
python main.py
```

## Model Architecture
The 1D CNN consists of:
- Multiple 1D Convolutional layers with ReLU activation.
- MaxPooling layers for downsampling features.
- Dropout layers for regularization (preventing overfitting).
- Dense (Fully Connected) layers for final emotion classification.

---
