import os
import json
import re
from geopy.geocoders import Nominatim
from unidecode import unidecode


# ------------ Lyrics ------------

# Read lyrics from files
lyrics = {}
for root, dirs, files in os.walk("data/kaggle-hip-hop"):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'rb') as f:
                name = file.split('_')[0]
                lyrics[name] = f.read().decode('utf-8', errors='ignore')

# Combine all lyrics
all_lyrics = "".join([f"|||{author}|||\n{text}\n" for author, text in lyrics.items()])

# Clean up lyrics
# Replace smart quotes and other special character with their ASCII equivalents
all_lyrics = unidecode(all_lyrics)
# all_lyrics = all_lyrics.replace("”", '"')
# all_lyrics = all_lyrics.replace("“", '"')
# all_lyrics = all_lyrics.replace("’", "'")
# all_lyrics = all_lyrics.replace("′", "'")
# all_lyrics = all_lyrics.replace("‘", "'")
# all_lyrics = all_lyrics.replace("´", "'")
# all_lyrics = all_lyrics.replace("—", "-")
# all_lyrics = all_lyrics.replace("–", "-")
# all_lyrics = all_lyrics.replace("…", "...")
# all_lyrics = all_lyrics.replace("•", "*")

# Remove messages from Genius
all_lyrics = all_lyrics.replace("Unfortunately, we are not licensed to display the full lyrics for this song at the moment. Hopefully we will be able to in the future. Until then... how about a random page?", "")

# Shorten any longer than 3 empty lines
all_lyrics = re.sub(r"\n{4,}", "\n\n\n", all_lyrics)

# Remove track listings
# Usually start with "Tracklist" followed by a numbered list of names, then end with something like "Cover Art:" or "Album Cover:"
all_lyrics = re.sub(r"Tracklist.*?\n?((\d+\.\s?.*|Deluxe:)\n+)+(.*?(Cover|Album) (Art|Artwork|Cover).*?\n)+", "", all_lyrics)

# Save to a single file
with open('data/all_lyrics.txt', 'w') as f:
    f.write(all_lyrics)


# ------------ Cities ------------

# Get cities
with open("data/cities.json", "r") as f:
    rapper_to_city = json.load(f)

# Load city coordinates
if os.path.exists("data/city_coords.json"):
    with open("data/city_coords.json", "r") as f:
        city_coords = json.load(f)
else:
    city_coords = {}

# Geolocate cities if not already done
geolocator = Nominatim(user_agent="rapgeo")
for city in set(rapper_to_city.values()) - set(city_coords.keys()):
    location = geolocator.geocode(city)
    if location:
        city_coords[city] = (location.latitude, location.longitude)

# Save to a file for future use
with open("data/city_coords.json", "w") as f:
    json.dump(city_coords, f)

