import joblib
import pandas as pd
from keras.models import load_model
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from nlp_features import extract_nlp_features
from train_model import preprocess_data, train_model
from graph_features import extract_graph_features_single_url
from train_model import custom_binary_crossentropy

# Load and preprocess the merged dataset
dataset_path = 'Merged_Dataset_with_Graph_and_NLP_Features.csv'
dataset = pd.read_csv(dataset_path)
X_train, X_val, y_train, y_val, scaler = preprocess_data(dataset)
print("Training dataset columns:\n", dataset.columns)

# Train the model
model = train_model(X_train, y_train, X_val, y_val)

# Save the trained model and preprocessing objects
model.save('trained_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Function to preprocess new URLs
def preprocess_new_url(url: str):
    graph_features = extract_graph_features_single_url(url)
    print("Graph Features:\n", graph_features)

    nlp_features = extract_nlp_features([url])
    print("NLP Features:\n", nlp_features)

    original_columns = dataset.drop(columns=['serial_number', 'URL', 'status']).columns
    combined_features = pd.DataFrame(columns=original_columns)
    combined_features.loc[0] = 0  # Initialize the row with zeros

    # Update graph features
    for col in graph_features.keys():
        if col in combined_features.columns:
            combined_features.at[0, col] = graph_features[col]

    # Update NLP features
    for col in nlp_features.columns:
        if col in combined_features.columns:
            combined_features.at[0, col] = nlp_features.at[0, col]

    combined_features['URL'] = url  # Add the URL column
    print("Combined Features:\n", combined_features)
    return combined_features

def pad_features_with_zeros(combined_features: pd.DataFrame, original_columns: pd.Index):
    # Initialize a DataFrame with zeros and original columns
    padded_features = pd.DataFrame(columns=original_columns)
    padded_features.loc[0] = 0

    # Fill the padded_features with values from combined_features
    for col in combined_features.columns:
        if col in padded_features.columns:
            padded_features.at[0, col] = combined_features.at[0, col]

    return padded_features

# Function to classify new URLs
def classify_new_url(url: str):
    features = preprocess_new_url(url)
    features = features.drop(columns=['URL'])  # Drop the URL column

    original_columns = dataset.drop(columns=['URL', 'status']).columns
    features_padded = pad_features_with_zeros(features, original_columns)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        features_scaled = scaler.transform(features_padded)

    prediction = model.predict(features_scaled)
    return prediction

# Example usage
model = load_model('trained_model.h5', custom_objects={'custom_binary_crossentropy': custom_binary_crossentropy})
scaler = joblib.load('scaler.pkl')

url = 'http://www.google.com'
prediction = classify_new_url(url)
classification_result = int(prediction[0][0])
if classification_result == 1:
    print(f"The URL {url} is classified as legitimate.")
else:
    print(f"The URL {url} is classified as phishing.")
