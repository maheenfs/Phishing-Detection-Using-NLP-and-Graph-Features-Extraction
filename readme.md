# **Phishing Detection Model using NLP and Graph Feature Extraction**

This project aims to build a phishing detection model using NLP and graph feature extraction techniques. The original dataset was taken from ZephyrCX's GitHub repository before extracting NLP and graph features.

## **Files**

**graph_features.py**

This script contains functions for extracting graph-based features from URLs in the dataset. It calculates various features related to the URL's structure, such as domain tokens, URL length, special characters, etc.

**nlp_features.py**

This script contains functions for extracting natural language processing (NLP) features from URLs in the dataset. It uses the nlp library to tokenize and vectorize the URL text, generating a fixed number of features for each URL.

**main.py**

This script is responsible for loading the dataset, extracting graph and NLP features, and merging them into a single dataset. The merged dataset is then saved as **Merged_Dataset_with_Graph_and_NLP_Features.csv**

**train_model.py**

This script trains a neural network model on the merged dataset (created by main.py). The script preprocesses the data, builds and compiles the model, and trains it using the dataset. It also evaluates the model on a validation set and reports performance metrics, such as accuracy, precision, recall, and F1-score. Finally, the trained model is saved as phishing_detection_model.h5.

**phishing_detection_pipeline.py**

This script contains a pipeline for detecting phishing websites using the trained neural network model. It loads and preprocesses the merged dataset, trains the model, and saves the model and preprocessing objects. The script also defines functions for preprocessing new URLs, padding features with zeros, and classifying new URLs as phishing or legitimate.

This is can be used for deploying the model in various ways but for this project I will not be deploying as Sir has told that it is not necessary for the project. I have included this as an extra to show that I have been working towards it.

## Usage

**Install the required dependencies:**

`pip install numpy pandas tensorflow keras scikit-learn joblib`

**Place the following files in the same directory:** 

graph_features.py, nlp_features.py, main.py, train_model.py, and phishing_detection_pipeline.py.

**Run main.py to generate the merged dataset with graph and NLP features:**

`python main.py`

(Warning: This takes a while to run. It Took me five hours. Run this if you have the time. I have already included the feature extracted dataset Merged_Dataset_with_Graph_and_NLP_Features.csv I hope you can directly use it by running train_model.py without having to run the main.py for extraction. Running phishing_detection_pipeline.py will result in the same long time needed.)

**Run train_model.py to train the neural network model:**

`python train_model.py`

Till this is enough to show the functioning of the project.

**Run phishing_detection_pipeline.py to use the pipeline for detecting phishing websites:**

`python phishing_detection_pipeline.py`