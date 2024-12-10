import os
import re
import json
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


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
    with open("data/all_lyrics_cleaned.txt", "r") as f:
        all_lyrics = f.read()

    # Split by author
    # Format is: |||Author|||\nText\n|||Author|||\nText\n...
    split = re.split(r"\|\|\|(.+?)\|\|\|\n", all_lyrics)[1:]
    texts_by_author = {author: text for author, text in zip(split[::2], split[1::2])}
    print(f"Number of Authors: {len(texts_by_author)}")

    # Collect embeddings
    embeddings = []
    labels = []
    for author, text in texts_by_author.items():
        embeddings.append(encode_text(text))
        labels.append(author)

    embeddings = np.array(embeddings)
    pickle.dump((embeddings, labels), open("embeddings.pkl", "wb"))


# Load new rappers' lyrics for testing
test_embeddings = []
test_labels = []
for file in os.listdir("data/test"):
    if file.endswith(".txt"):
        with open(f"data/test/{file}", "r") as f:
            test_text = f.read()
            test_embeddings.append(encode_text(test_text))
            test_labels.append(file[:-4])

test_embeddings = np.array(test_embeddings)

# Get cities and their coordinates (created in prepare.py)
with open("data/cities.json", "r") as f:
    rapper_to_city = json.load(f)
with open("data/city_coords.json", "r") as f:
    city_coords = json.load(f)

# Map rappers to city coordinates
rapper_coords = {rapper: city_coords[city] for rapper, city in rapper_to_city.items() if city in city_coords}

# Convert to a numpy array
coords_array = np.array(list(rapper_coords.values()))  # Shape: (num_rappers, 2)






# ------------ Map Function ------------

def plot_map(coords, predicted_coords, labels, title):
    # Create a map
    plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-66, resolution='i')

    # Draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    # Plot cities
    at_location = set()
    for i, rapper in enumerate(labels):
        # Get the predicted coordinates
        lat_pred, lon_pred = predicted_coords[i]
        x_pred, y_pred = m(lon_pred, lat_pred)

        # Avoid overlap by incrementing the y-coordinate
        while (x_pred, y_pred) in at_location:
            y_pred += 40000
        at_location.add((x_pred, y_pred))

        # Plot the predicted location
        plt.plot(x_pred, y_pred, 'ro', markersize=2)
        plt.text(x_pred, y_pred, rapper + " (predicted)", fontsize=3, color='red')

        if i < len(coords):
            # Get the actual coordinates
            lat, lon = coords[i]
            x, y = m(lon, lat)
            
            # Avoid overlap by incrementing the y-coordinate
            while (x, y) in at_location:
                y += 40000
            at_location.add((x, y))

            # Plot the actual location
            plt.plot(x, y, 'bo', markersize=2)
            plt.text(x, y, rapper, fontsize=3, color='blue')

            # Draw a line between the actual and predicted locations
            plt.plot([x, x_pred], [y, y_pred], 'r-', linewidth=0.5)

    plt.title(title)
    plt.show()

