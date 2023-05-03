import tensorflow as tf
import pandas as pd
import numpy as np
import os

SPLIT_TRAIN_TEST_AND_EVALUATE = False
CONVERT_TO_TFLITE = True

with open("out.csv", "r") as file:
    features = []
    labels = []
    for line in file:
        asList = line.split(",")
        features.append(asList[:-1])
        labels.append(asList[-1].strip())

    for i, label in enumerate(labels):
        labels[i] = int(label)-1

    features = pd.DataFrame(features).to_numpy()
    labels = np.array(labels)

    labels = labels.astype(int)
    features = features.astype(float)

    # shuffle
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    # normalize
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    if SPLIT_TRAIN_TEST_AND_EVALUATE:
        # split into train and test
        train_size = int(len(features) * 0.8)
        test_size = len(features) - train_size
        train_features, test_features = features[0:train_size], features[train_size:len(features)]
        train_labels, test_labels = labels[0:train_size], labels[train_size:len(labels)]
    else:
        train_features, test_features = features, features
        train_labels, test_labels = labels, labels

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
        tf.keras.layers.Dense(3, activation='sigmoid')
    ])

    # Compile the model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    # Train the model.
    model.fit(train_features, train_labels, epochs=40, batch_size=32)

    if SPLIT_TRAIN_TEST_AND_EVALUATE:
        # Evaluate the model.
        print("Evaluating model...")
        model.evaluate(test_features, test_labels)

    if CONVERT_TO_TFLITE:
        os.system("rm -rf model")
        os.system("rm -f model_tflite")

        model.save("model", save_format="tf")

        converter = tf.lite.TFLiteConverter.from_saved_model("model")
        tflite_model = converter.convert()

        open("model_tflite", "wb").write(tflite_model)

        os.system("xxd -i model_tflite > model_tflite.h")