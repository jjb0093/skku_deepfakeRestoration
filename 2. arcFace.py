import cv2, os, pickle
from insightface.app import FaceAnalysis

import numpy as np

def getIdentity(path):
    img = cv2.imread(path)
    faces = app.get(img)

    if(len(faces) == 0):
        return None

    faces = sorted(faces, key = lambda f: f.bbox[2] * f.bbox[3], reverse = True)
    emb = faces[0].normed_embedding

    return emb

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

path = r"Data/val"
folders = os.listdir(path)

for folder in folders:
    folderPath = f"{path}\{folder}"
    files = os.listdir(folderPath)

    embFolderPath = f"Embedding/val/{folder}"
    os.makedirs(embFolderPath, exist_ok = True)

    for file in files:
        emb = getIdentity(f"{folderPath}/{file}")
        if(emb is None): continue

        print(f"{folderPath}/{file}")
        #print(emb, end = "\n\n")
        
        with open(f"{embFolderPath}/{file.replace('.jpg', '')}.pkl", 'wb') as f:
            pickle.dump(emb, f)
