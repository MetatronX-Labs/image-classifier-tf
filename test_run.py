
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

# Load pre-trained model
model = load_model('models/model_fashion.h5')

# Load test data
(_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Evaluate model
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {acc * 100:.2f}%")
