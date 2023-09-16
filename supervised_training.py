from nn_builder.pytorch.NN import NN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

print(model)

# Freeze all layers except the last output layer
for name, param in model.named_parameters():
    if not name.startswith('output_layers.0'):
        print("Freezing layer", name)
        param.requires_grad = False

# Define parameters for the dataset
input_matrix_size = (12, 12)  # Size of each 6x6 matrix
num_classes = 4  # Number of possible classes
learning_rate = 0.01
batch_size = 55
num_epochs = 10000
hidden_size = 64

# Create an instance of the model
# model = SimpleNN(input_size=np.prod(input_matrix_size), hidden_size=hidden_size, num_classes=num_classes)

unsafe_transitions = np.load('unsafe_transitions.npy', allow_pickle=True)

dataset = np.load('safe-grid-gym/dataset_supervised.npy', allow_pickle=True)

added_unsafe = 0
for unsafe_sample in unsafe_transitions:
    found = False
    for sample in dataset:
        if unsafe_sample['action'] == sample['action'] and np.array_equal(unsafe_sample['state'], sample['state']):
            found = True
    if not found:
        dataset = np.append(dataset, unsafe_sample)
        dataset = np.append(dataset, unsafe_sample)
        dataset = np.append(dataset, unsafe_sample)
        dataset = np.append(dataset, unsafe_sample)
        dataset = np.append(dataset, unsafe_sample)
        added_unsafe += 5

# Generate random input data (6x6 matrices)
X_train = [x['state'] for x in dataset]

# Generate random class labels (0 or 1) for each class and sample
action_labels = [x['action'] for x in dataset]
one_hot_encodings = np.eye(num_classes)[action_labels]
y_train = [x if i in range(len(one_hot_encodings) - added_unsafe) else 1 - x for i, x in enumerate(one_hot_encodings)]

print(y_train)
# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

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
        predicted = torch.round(outputs)  # Convert logits to binary predictions (0 or 1)
        total += batch_labels.size(0)
        print(predicted)
        correct += (predicted == batch_labels).all(dim=1).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True, min_lr=1e-8)

# Training loop
for epoch in range(num_epochs):
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

    # Calculate the average loss for this epoch
    avg_loss = total_loss / len(train_loader)

    # Update the learning rate scheduler with the current loss
    scheduler.step(total_loss)

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, LR: {optimizer.param_groups[0]['lr']}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        outputs = model(batch_inputs)
        # print(outputs)
        predicted = torch.round(outputs)  # Convert logits to binary predictions (0 or 1)
        total += batch_labels.size(0)
        check = (predicted == batch_labels).all(dim=1).sum().item()
        print(batch_labels, predicted, check)
        correct += check

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), './Models/supervised_model.pt')
