import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data_prep import load_ravdess_data
from src.model import build_model

def main():
    print("--- Emotion Recognition from Speech Project ---")
    
    data_path = 'data/ravdess'
    
    # 1. Load Data
    print(f"Loading RAVDESS dataset from {data_path}...")
    X, y = load_ravdess_data(data_path)
    
    if X is None or len(X) == 0:
        print("Error: No RAVDESS data found. Please ensure .wav files are in data/ravdess/")
        return
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # 3. Build Model
    model = build_model(input_shape=40, num_classes=8)
    
    # 4. Train Model
    print("\nStarting training...")
    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=16, 
                        validation_data=(X_test, y_test),
                        verbose=1)
    
    # 5. Save Results
    print("\nSaving results...")
    os.makedirs('outputs', exist_ok=True)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy_plot.png')
    plt.close()
    
    print("Accuracy plot saved to outputs/accuracy_plot.png")
    print("\n--- Pipeline Execution Complete ---")
    print("Model trained and evaluated on RAVDESS features.")
    print("Check 'outputs/' for performance visualizations.")

if __name__ == "__main__":
    main()
