import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (BatchNormalization, Conv1D, Dense, Dropout, Flatten,
                          Lambda, LeakyReLU, MaxPooling1D)
from keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from tensorflow.debugging import assert_all_finite

# Constants
EPSILON = 1e-8
LEARNING_RATE = 0.00001
CLIPNORM = 1.0
BATCH_SIZE = 32
EPOCHS = 10

def load_dataset(file_path):
    """Load dataset from the specified file path."""
    return pd.read_csv(file_path)

def check_for_invalid_values(dataset):
    """Check if the dataset contains NaN or infinite values."""
    return dataset.isnull().values.any() or np.isinf(dataset.values).any()

def preprocess_data(dataset):
    """Preprocess the dataset and split it into training and validation sets."""
    X = dataset.drop(columns=['URL', 'status'])
    y = dataset['status']

    # Fill NaN values with the mean value of each feature
    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Add epsilon to avoid division by zero errors
    X_train = X_train + EPSILON
    X_val = X_val + EPSILON

    # Expand dimensions to fit the input shape of the model
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)

    return X_train, X_val, y_train, y_val, scaler

def custom_binary_crossentropy(y_true, y_pred):
    """Custom binary cross-entropy loss function with NaN and infinite values check."""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = K.mean(-y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=-1)
    assert_all_finite(loss, message="Loss contains NaN or infinite values.")
    return loss

def build_and_compile_model(input_shape):
    """Build and compile the neural network model."""
    model = Sequential()

    # Convolutional layers
    model.add(Conv1D(filters=64, kernel_size=3, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, kernel_size=3))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.add(Lambda(lambda x: x + EPSILON))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)
    model.compile(optimizer=optimizer, loss=custom_binary_crossentropy, metrics=['accuracy'])

    return model


def train_model(X_train, y_train, X_val, y_val):
    """Train the model using the training dataset."""
    model = build_and_compile_model(X_train.shape[1:])
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
                        verbose=1)
    return model


def evaluate_and_report_metrics(y_true, y_pred):
    """Evaluate the model and report performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset('Merged_Dataset_with_Graph_and_NLP_Features.csv')

    if check_for_invalid_values(dataset):
        print("The dataset contains NaN or infinite values. Please handle them before proceeding.")

    print("Preprocessing data...")
    X_train, X_val, y_train, y_val, scaler = preprocess_data(dataset)

    print("Training the model...")
    model = train_model(X_train, y_train, X_val, y_val)

    print("Predicting on the validation set...")
    y_val_pred = np.round(model.predict(X_val) + EPSILON).flatten()

    print("Evaluating and reporting metrics...")
    evaluate_and_report_metrics(y_val, y_val_pred)

    # Save the model if desired
    model.save("phishing_detection_model.h5")

