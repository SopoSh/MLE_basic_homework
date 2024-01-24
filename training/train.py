import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset

# Function to load the Iris dataset from Wikipedia
def load_iris_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

# Function to preprocess the data and split it into training and inference sets
def preprocess_data(data):
    X = data.iloc[:, :-1].values
    y = data['target'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and inference sets
    X_train, X_inference, y_train, y_inference = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_inference, y_inference

# Define a simple neural network model using PyTorch
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

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == labels).float().mean()
    print(f'Accuracy on the inference set: {accuracy.item()}')

# Unit tests can be added based on specific requirements

if __name__ == "__main__":
    # Load and preprocess the Iris dataset
    iris_data = load_iris_data()
    X_train, y_train, X_inference, y_inference = preprocess_data(iris_data)

    # Create and train the model
    input_size = X_train.shape[1]
    output_size = len(set(y_train))
    model = SimpleNN(input_size, hidden_size=8, output_size=output_size)
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_inference, y_inference)
