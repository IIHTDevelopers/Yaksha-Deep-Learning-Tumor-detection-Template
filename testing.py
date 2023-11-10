import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset, tf_parse

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def make_predictions(data_dir):
    path = data_dir
    batch_size = 8
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x) // batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'accuracy': tf.keras.metrics.binary_accuracy}):
        model = tf.keras.models.load_model("files/model1.keras")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.binary_accuracy])

    accuracy = model.evaluate(test_dataset, steps=test_steps)[-1]

    for i, (x, y) in enumerate(zip(test_x, test_y)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        y_color = cv2.cvtColor((y * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        y_pred_color = cv2.cvtColor((y_pred * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        all_images = [
            (x * 255.0).astype(np.uint8), white_line.astype(np.uint8),
            y_color, white_line.astype(np.uint8),
            y_pred_color
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)

    if accuracy < 0.9:
        print("Model failed.")
    
    else:
        print("Model passed.")

if __name__ == "__main__":
    make_predictions("test-images/")
