import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def extract_mfcc(file_path):
    """
    Extracts MFCC features from an audio file.
    """
    try:
        # Load audio file (resample to 22050 Hz and convert to mono)
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs (40 features)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_ravdess_data(data_path):
    """
    Loads RAVDESS dataset and extracts labels from filenames.
    Filename example: 03-01-01-01-01-01-01.wav (Index 2 is emotion)
    """
    features = []
    labels = []
    
    # Check if directory exists
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found.")
        return None, None

    for root, dirs, files in os.walk(data_path):
        for file in tqdm(files, desc="Processing audio files"):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Extract emotion label
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion = int(parts[2])
                    mfccs = extract_mfcc(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        labels.append(emotion - 1) # 0-indexed labels
                
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Example usage (would run if data existed)
    # X, y = load_ravdess_data('data/ravdess')
    print("Data prep module ready.")
