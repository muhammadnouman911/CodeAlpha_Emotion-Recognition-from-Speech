import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    """
    Builds a 1D CNN model for audio classification.
    """
    model = models.Sequential([
        # Expansion layer to add channel dimension for 1D CNN
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model(40, 8) # 40 MFCCs, 8 emotions
    model.summary()
