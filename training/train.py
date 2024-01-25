import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import logging
import argparse
import json
import pickle
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir


# Load environment variables from .env file
load_dotenv()

CONF_FILE = os.getenv('CONF_PATH')
# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")

import mlflow
mlflow.autolog()

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Function to evaluate the model
def evaluate_model(model, X_inference, y_inference):
    inputs = torch.tensor(X_inference, dtype=torch.float32)
    labels = torch.tensor(y_inference, dtype=torch.long)

    if not hasattr(model, 'forward'):
        raise ValueError("The provided model does not have a forward method. Make sure it is a valid PyTorch model.")

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == labels).float().mean()
    print(f'Accuracy on the inference set: {accuracy.item()}')


def save(path: str, model) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
        else:
            path = os.path.join(MODEL_DIR, path)

        if not hasattr(model, 'state_dict'):
            raise ValueError("The provided model does not have a state_dict method. It may not be serializable.")

        with open(path, 'wb') as f:
            pickle.dump(model, f)


from data_process.data_generation import preprocess_data


if __name__ == "__main__":
    try:
        # Load and preprocess the Iris dataset
        X_train, y_train, X_inference, y_inference = preprocess_data()

        # Create and train the model
        input_size = X_train.shape[1]
        output_size = len(set(y_train))
        model = SimpleNN(input_size, hidden_size=8, output_size=output_size)
        train_model(model, X_train, y_train)

        # Save the model
        save(path=None, model=model)

        # Evaluate the model
        evaluate_model(model, X_inference, y_inference)
    except Exception as e:
        print(f"An error occurred: {e}")
