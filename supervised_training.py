from nn_builder.pytorch.NN import NN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
device = "cpu"

world_shape = (12, 12)

hyperparameters = {
    "input_dim": world_shape[0] * world_shape[1],
    "output_dim": 4,
    "linear_hidden_units": [128, 128, 64],
    "initialiser": "Xavier"
}

model = NN(input_dim=world_shape[0] * world_shape[1],
                   layers_info=hyperparameters["linear_hidden_units"] + [hyperparameters["output_dim"]],
                   initialiser=hyperparameters["initialiser"],
                   random_seed=42, output_activation="sigmoid").to(device)

model_name = "SAC_Discrete_local_network.pt"
model.load_state_dict(torch.load('./Models/' + model_name))

unsafe_transitions = np.load('unsafe_transitions.npy', allow_pickle=True)


# Define parameters for the dataset
num_samples = 1000  # Number of data samples
input_matrix_size = (12, 12)  # Size of each 6x6 matrix
num_classes = 4  # Number of possible classes
hidden_size = 64  # Size of the hidden layer
learning_rate = 0.0001
batch_size = 32
num_epochs = 1000 * 3

# Generate random input data (6x6 matrices)
X_train = [x['state'] for x in unsafe_transitions]

# Generate random class labels (0 or 1) for each class and sample
action_labels = [x['action'] for x in unsafe_transitions]
one_hot_encodings = np.eye(num_classes)[action_labels]
y_train = 1 - one_hot_encodings

print(f"{y_train=}")

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        outputs = model(batch_inputs)
        print(outputs)
        predicted = torch.round(outputs) # Convert logits to binary predictions (0 or 1)
        total += batch_labels.size(0)
        print(predicted)
        correct += (predicted == batch_labels).all(dim=1).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Training loop
for epoch in range(num_epochs):
    if epoch % 300 == 0:
        learning_rate /= 2
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)  # Flatten input
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, LR: {learning_rate}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        outputs = model(batch_inputs)
        print(outputs)
        predicted = torch.round(outputs) # Convert logits to binary predictions (0 or 1)
        total += batch_labels.size(0)
        print(predicted)
        correct += (predicted == batch_labels).all(dim=1).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), 'supervised_model.pt')