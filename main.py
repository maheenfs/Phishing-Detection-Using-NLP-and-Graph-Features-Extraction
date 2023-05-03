import pandas as pd
from graph_features import extract_graph_features
from nlp_features import extract_nlp_features

# Load the dataset
print("Loading dataset...")
dataset_path = 'Dataset.csv'
dataset = pd.read_csv(dataset_path)
urls = dataset['URL'].tolist()


# Extract graph-based features
print("Extracting graph-based features...")
graph_features = extract_graph_features(dataset_path)

# Extract NLP-based features
print("Extracting NLP-based features...")
nlp_features = extract_nlp_features(urls, max_features=100, max_workers=8, batch_size=1000)


# Combine the original dataset with the extracted features
print("Merging datasets...")
merged_dataset = pd.concat([dataset, graph_features.drop(columns=['URL']), nlp_features], axis=1)

# Save the merged dataset to a CSV file
print("Saving merged dataset to file...")
merged_dataset.to_csv('Merged_Dataset_with_Graph_and_NLP_Features.csv', index=False)

print("Done!")
