import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import json
import pickle
import os
from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load a pre-trained language model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Encode each text to a vector
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # Mean pooling of embeddings for sentence representation
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
    return embeddings[0]


# Load embeddings and labels
if os.path.exists("embeddings.pkl") and (saved := pickle.load(open("embeddings.pkl", "rb"))):
    embeddings, labels = saved
else:
    # Sample texts by authors
    with open("kaggle-hip-hop/lyrics.json", "r") as f:
        texts_by_author = json.load(f)

    # Collect embeddings
    embeddings = []
    labels = []
    for author, text in texts_by_author.items():
        embeddings.append(encode_text(text))
        labels.append(author)

    embeddings = np.array(embeddings)
    pickle.dump((embeddings, labels), open("embeddings.pkl", "wb"))


# Get cities
with open("kaggle-hip-hop/cities.json", "r") as f:
    rapper_to_city = json.load(f)

# Load city coordinates
if os.path.exists("city_coords.json"):
    with open("city_coords.json", "r") as f:
        city_coords = json.load(f)
else:
    # Geolocate cities
    geolocator = Nominatim(user_agent="rapgeo")
    city_coords = {}
    for city in set(rapper_to_city.values()):
        location = geolocator.geocode(city)
        if location:
            city_coords[city] = (location.latitude, location.longitude)

    # Save to a file for future use
    with open("city_coords.json", "w") as f:
        json.dump(city_coords, f)

# Map rappers to city coordinates
rapper_coords = {rapper: city_coords[city] for rapper, city in rapper_to_city.items() if city in city_coords}

# Convert to a numpy array
coords_array = np.array(list(rapper_coords.values()))  # Shape: (num_rappers, 2)


# ------------ PCA and Linear Regression -> Map ------------

if False:
    # Apply PCA to reduce to 10 components
    pca = PCA(n_components=10)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Multivariate regression for both latitude and longitude
    multi_model = MultiOutputRegressor(LinearRegression())
    multi_scores = cross_val_score(multi_model, reduced_embeddings, coords_array, cv=5, scoring='r2')
    print("Cross-validated R^2 for Latitude and Longitude (PCA):", multi_scores.mean())

    # Visualize Regression Results
    multi_model.fit(reduced_embeddings, coords_array)
    predicted_coords = multi_model.predict(reduced_embeddings)

    # Create a map
    plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-66, resolution='i')

    # Draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    # Plot cities
    num_at_location = {}
    for rapper, (lat, lon) in rapper_coords.items():
        x, y = m(lon, lat)

        # Avoid overlap by incrementing the y-coordinate
        num = num_at_location.get((x, y), 0)
        num_at_location[(x, y)] = num + 1
        y += num * 50000

        x_pred, y_pred = m(predicted_coords[labels.index(rapper), 1], predicted_coords[labels.index(rapper), 0])

        plt.plot(x, y, 'bo', markersize=2)
        plt.plot(x_pred, y_pred, 'ro', markersize=2)
        plt.text(x, y, rapper, fontsize=3, color='blue')
        plt.text(x_pred, y_pred, rapper + " (predicted)", fontsize=3, color='red')

    plt.title("Geographic Distribution of Rappers")
    plt.show()



# ------------ PCA and Random Forest -> Map ------------

# Perform PCA
pca = PCA()
pca.fit(embeddings)

# Scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
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

# Random Forest for both latitude and longitude
forest_model = MultiOutputRegressor(RandomForestRegressor())

# Grid search for hyperparameters
print("Performing Grid Search for Random Forest Hyperparameters...")
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [5, 10, 20, None]
}
grid_search = GridSearchCV(forest_model, param_grid, cv=5, scoring='r2')
grid_search.fit(reduced_embeddings, coords_array)
print("Best Parameters:", grid_search.best_params_)
forest_model = grid_search.best_estimator_

# Cross-validated R^2
forest_scores = cross_val_score(forest_model, reduced_embeddings, coords_array, cv=5, scoring='r2')
print("Cross-validated R^2 for Latitude and Longitude (PCA):", forest_scores.mean())

# Visualize Random Forest Results
forest_model.fit(reduced_embeddings, coords_array)
predicted_coords = forest_model.predict(reduced_embeddings)

# Create a map
plt.figure(figsize=(12, 8))
m = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-66, resolution='i')

# Draw map features
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Plot cities
num_at_location = {}
for rapper, (lat, lon) in rapper_coords.items():
    x, y = m(lon, lat)

    # Avoid overlap by incrementing the y-coordinate
    num = num_at_location.get((x, y), 0)
    num_at_location[(x, y)] = num + 1
    y += num * 50000

    x_pred, y_pred = m(predicted_coords[labels.index(rapper), 1], predicted_coords[labels.index(rapper), 0])

    plt.plot(x, y, 'bo', markersize=2)
    plt.plot(x_pred, y_pred, 'ro', markersize=2)
    plt.text(x, y, rapper, fontsize=3, color='blue')
    plt.text(x_pred, y_pred, rapper + " (predicted)", fontsize=3, color='red')

    # Draw a line between the actual and predicted locations
    plt.plot([x, x_pred], [y, y_pred], 'k-', linewidth=0.5, color='red')

# Test on a new rapper
# Load a new rapper's lyrics
with open("test/Diddy.txt", "r") as f:
    text = f.read()

# Encode the text
new_embedding = encode_text(text)

# Reduce to 10 components
new_embedding = pca.transform(new_embedding.reshape(1, -1))

# Predict the location
predicted_coords = forest_model.predict(new_embedding)

# Plot the new rapper
x, y = m(predicted_coords[0, 1], predicted_coords[0, 0])
plt.plot(x, y, 'go', markersize=5)
plt.text(x, y, "Diddy", fontsize=5, color='green')


plt.title("Geographic Distribution of Rappers")
plt.show()



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
