import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import pickle
import os
import re
from mpl_toolkits.basemap import Basemap
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.linear_model import ElasticNetCV
from util import *

# ------------ Hierarchical Classification ------------

# Load the region classes
# Format: {"Rapper": ["East Coast", "New York", "Brooklyn"]}
with open("data/regions.json", "r") as f:
    rapper_to_classes = json.load(f)

# Map authors (rappers) to their hierarchical labels
region_labels = []
state_labels = []
city_labels = []

for label in labels:
    if label in rapper_to_classes:
        region, state, city = rapper_to_classes[label]
        region_labels.append(region)
        state_labels.append(state)
        city_labels.append(city)
    else:
        print(f"Warning: {label} not found in region mapping.")

# Encode hierarchical labels
region_encoder = LabelEncoder()
state_encoder = LabelEncoder()
city_encoder = LabelEncoder()

region_labels_encoded = region_encoder.fit_transform(region_labels)
state_labels_encoded = state_encoder.fit_transform(state_labels)
city_labels_encoded = city_encoder.fit_transform(city_labels)

# Split data
X_train, X_test, y_region_train, y_region_test = train_test_split(
    embeddings, region_labels_encoded, test_size=0.2, random_state=42
)
_, _, y_state_train, y_state_test = train_test_split(
    embeddings, state_labels_encoded, test_size=0.2, random_state=42
)
_, _, y_city_train, y_city_test = train_test_split(
    embeddings, city_labels_encoded, test_size=0.2, random_state=42
)

# Convert labels to tensors
y_region_train = torch.tensor(y_region_train, dtype=torch.long)
y_state_train = torch.tensor(y_state_train, dtype=torch.long)
y_city_train = torch.tensor(y_city_train, dtype=torch.long)

# Define hierarchical model
class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, num_regions, num_states, num_cities):
        super(HierarchicalClassifier, self).__init__()
        # Shared layers
        self.shared_layer = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.3)
        
        # Region classifier
        self.region_head = nn.Linear(256, num_regions)
        
        # State classifier (conditioned on region)
        self.state_head = nn.Linear(256 + num_regions, num_states)
        
        # City classifier (conditioned on region and state)
        self.city_head = nn.Linear(256 + num_regions + num_states, num_cities)

    def forward(self, x):
        # Shared representation
        shared_output = self.dropout(torch.relu(self.shared_layer(x)))
        
        # Region prediction
        region_logits = self.region_head(shared_output)
        region_probs = torch.softmax(region_logits, dim=1)
        
        # Concatenate region probabilities
        region_features = torch.cat([shared_output, region_probs], dim=1)
        
        # State prediction
        state_logits = self.state_head(region_features)
        state_probs = torch.softmax(state_logits, dim=1)
        
        # Concatenate state probabilities
        state_features = torch.cat([region_features, state_probs], dim=1)
        
        # City prediction
        city_logits = self.city_head(state_features)
        
        return region_logits, state_logits, city_logits

# Initialize model
num_regions = len(region_encoder.classes_)
num_states = len(state_encoder.classes_)
num_cities = len(city_encoder.classes_)
print(f"Number of Regions: {num_regions}, States: {num_states}, Cities: {num_cities}")

model = HierarchicalClassifier(input_dim=embeddings.shape[1], num_regions=num_regions, num_states=num_states, num_cities=num_cities)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Includes L2 regularization

# Scheduler for learning rate decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

num_epochs = 500
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    region_loss_sum = 0
    state_loss_sum = 0
    city_loss_sum = 0

    for i in range(0, len(X_train_tensor), batch_size):
        # Batch data
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y_region = y_region_train[i:i+batch_size]
        batch_y_state = y_state_train[i:i+batch_size]
        batch_y_city = y_city_train[i:i+batch_size]

        # Forward pass
        region_logits, state_logits, city_logits = model(batch_X)
        
        # Compute weighted losses
        loss_region = criterion(region_logits, batch_y_region)
        loss_state = criterion(state_logits, batch_y_state)
        loss_city = criterion(city_logits, batch_y_city)
        
        # Weighted combination of losses
        alpha_region, alpha_state, alpha_city = 0.5, 0.3, 0.2
        loss = alpha_region * loss_region + alpha_state * loss_state + alpha_city * loss_city
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        region_loss_sum += loss_region.item()
        state_loss_sum += loss_state.item()
        city_loss_sum += loss_city.item()

    # Update learning rate
    scheduler.step()
    
    # Print epoch metrics
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.4f}, "
        f"Region Loss: {region_loss_sum:.4f}, State Loss: {state_loss_sum:.4f}, City Loss: {city_loss_sum:.4f}"
    )

# Evaluate the model
model.eval()
with torch.no_grad():
    region_logits, state_logits, city_logits = model(X_test_tensor)
    
    # Predictions
    region_preds = torch.argmax(region_logits, dim=1).numpy()
    state_preds = torch.argmax(state_logits, dim=1).numpy()
    city_preds = torch.argmax(city_logits, dim=1).numpy()

    # Accuracy
    region_accuracy = accuracy_score(y_region_test, region_preds)
    state_accuracy = accuracy_score(y_state_test, state_preds)
    city_accuracy = accuracy_score(y_city_test, city_preds)

print("Region Accuracy:", region_accuracy)
print("State Accuracy:", state_accuracy)
print("City Accuracy:", city_accuracy)

# Test new rappers
test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32)

with torch.no_grad():
    region_logits, state_logits, city_logits = model(test_embeddings_tensor)
    
    # Predictions
    region_preds = torch.argmax(region_logits, dim=1).numpy()
    state_preds = torch.argmax(state_logits, dim=1).numpy()
    city_preds = torch.argmax(city_logits, dim=1).numpy()

    # Decode labels
    region_preds = region_encoder.inverse_transform(region_preds)
    state_preds = state_encoder.inverse_transform(state_preds)
    city_preds = city_encoder.inverse_transform(city_preds)

    for rapper, region, state, city in zip(test_labels, region_preds, state_preds, city_preds):
        print(f"{rapper}: {region}, {state}, {city}")

