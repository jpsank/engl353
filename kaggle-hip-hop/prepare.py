import os
import json

data = {}

for root, dirs, files in os.walk("lyrics"):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'rb') as f:
                name = file.split('_')[0]
                data[name] = f.read().decode('utf-8', errors='ignore')

with open('lyrics.json', 'w') as f:
    json.dump(data, f)
