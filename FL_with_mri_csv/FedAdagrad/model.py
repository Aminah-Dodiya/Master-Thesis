# import random
import tensorflow as tf
import keras

# # Set random seeds for reproducibility
# random.seed(64)
# np.random.seed(64)
# tf.random.set_seed(64)
 
# Dfine the neural network architecture
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=100),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu' ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu' ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    sgd = keras.optimizers.SGD()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model
