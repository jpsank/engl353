import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.linear_model import ElasticNetCV
from util import *


# ------------ ElasticNet Lasso/Ridge Regression -> Map ------------

# ElasticNet Regression for feature selection
lasso_model = MultiOutputRegressor(ElasticNetCV(cv=3))

# Grid search for hyperparameters
print("Performing Grid Search for ElasticNet Hyperparameters...")
param_grid = {
    'estimator__l1_ratio': [0.1, 0.5, 0.9],
    'estimator__alphas': [[0.02, 0.03, 0.04, 0.05, 0.1, 1.0, 10.0]]
}
grid_search = GridSearchCV(lasso_model, param_grid, cv=3, scoring='r2')
grid_search.fit(embeddings, coords_array)
print("Best Parameters:", grid_search.best_params_)
lasso_model = grid_search.best_estimator_

# Select the non-zero coefficients
nonzero_indices = np.nonzero(lasso_model.estimators_[0].coef_)[0]
nonzero_embeddings = embeddings[:, nonzero_indices]
print("Non-zero Coefficients:", nonzero_embeddings.shape[1])

# Cross-validated R^2
scores = cross_val_score(lasso_model, embeddings, coords_array, cv=5, scoring='r2')
print("Cross-validated R^2 for Latitude and Longitude (ElasticNet for Regression):", scores.mean())

# Fit the model
lasso_model.fit(embeddings, coords_array)
predicted_coords = lasso_model.predict(embeddings)

# Test on a new rapper
new_predicted_coords = lasso_model.predict(test_embeddings)

# Plot the actual and predicted locations
plot_map(coords_array, np.vstack([predicted_coords, new_predicted_coords]), labels + test_labels, 
         "Geographic Distribution of Rappers (ElasticNet for Regression)")






# ------------ Lasso for Feature Selection + Random Forest Regression -> Map ------------

# Random Forest Regressor
rf_model = MultiOutputRegressor(RandomForestRegressor())

# Grid search for hyperparameters
print("Performing Grid Search for Random Forest Hyperparameters...")
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [5, 10, 20, None]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2')
grid_search.fit(nonzero_embeddings, coords_array)
print("Best Parameters:", grid_search.best_params_)
rf_model = grid_search.best_estimator_

# Cross-validated R^2
scores = cross_val_score(rf_model, nonzero_embeddings, coords_array, cv=5, scoring='r2')
print("Cross-validated R^2 for Latitude and Longitude (Lasso Feature Selection + Random Forest):", scores.mean())

# Fit the model
rf_model.fit(nonzero_embeddings, coords_array)
predicted_coords = rf_model.predict(nonzero_embeddings)

# Test on new rappers
new_predicted_coords = rf_model.predict(test_embeddings[:, nonzero_indices])

# Plot the actual and predicted locations
plot_map(coords_array, np.vstack([predicted_coords, new_predicted_coords]), labels + test_labels, 
         "Geographic Distribution of Rappers (Lasso Feature Selection + Random Forest)")




# ------------ PCA + Random Forest Regression -> Map ------------

# Perform PCA
pca = PCA()
pca.fit(embeddings)

if False:
    # Scree plot
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title("Scree Plot")
    plt.show()

# Automatic selection using the elbow method
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance > 0.95) + 1
print("Number of Components for 95% Variance:", n_components)

# Apply PCA to reduce to n components
pca = PCA(n_components=n_components)
reduced_embeddings = pca.fit_transform(embeddings)

# Random Forest Regressor
regress_model = MultiOutputRegressor(RandomForestRegressor())

# Grid search for hyperparameters
print("Performing Grid Search for Random Forest Hyperparameters...")
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [5, 10, 20, None]
}
grid_search = GridSearchCV(regress_model, param_grid, cv=5, scoring='r2')
grid_search.fit(reduced_embeddings, coords_array)
print("Best Parameters:", grid_search.best_params_)
regress_model = grid_search.best_estimator_

# Cross-validated R^2
scores = cross_val_score(regress_model, reduced_embeddings, coords_array, cv=5, scoring='r2')
print("Cross-validated R^2 for Latitude and Longitude (PCA + Random Forest):", scores.mean())

# Fit the model
regress_model.fit(reduced_embeddings, coords_array)
predicted_coords = regress_model.predict(reduced_embeddings)

# Test on a new rapper
new_predicted_coords = regress_model.predict(pca.transform(test_embeddings))

# Plot the actual and predicted locations
plot_map(coords_array, np.vstack([predicted_coords, new_predicted_coords]), labels + test_labels, "Geographic Distribution of Rappers (PCA + Random Forest)")








# ------------ PCA and 3D Plot ------------

# Apply PCA to reduce to 3 components
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# Use the embedding norm for color
color_map = np.linalg.norm(embeddings, axis=1)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
                     c=color_map, cmap='viridis', s=50)

# Add labels
for i, label in enumerate(labels):
    ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], label)

# Customize plot
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title("3D PCA of Author Texts with Embedding-Based Colors")
fig.colorbar(scatter, ax=ax, label="Embedding Norm")

plt.show()
