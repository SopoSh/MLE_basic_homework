# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Function to load the Iris dataset from Wikipedia
def load_iris_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

# Function to preprocess the data and split it into training and inference sets
def preprocess_data(data = load_iris_data()):
    X = data.iloc[:, :-1].values
    y = data['target'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and inference sets
    X_train, X_inference, y_train, y_inference = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine features and labels using pandas.concat
    train_df = pd.DataFrame(data=np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1), columns=data.columns)
    inference_df = pd.DataFrame(data=np.concatenate((X_inference, y_inference.reshape(-1, 1)), axis=1),
                                columns=data.columns)

    logger.info(f"Saving train data to {TRAIN_PATH}...")
    train_df.to_csv(TRAIN_PATH, index=False)

    logger.info(f"Saving inference data to {INFERENCE_PATH}...")
    inference_df.to_csv(INFERENCE_PATH, index=False)

    return X_train, y_train, X_inference, y_inference

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    preprocess_data(load_iris_data())
    logger.info("Script completed successfully.")