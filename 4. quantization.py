import os, pickle
import numpy as np
from time import sleep

def quantifier(x, c = 0.25):
    x = np.asarray(x, dtype = np.float32)
    x = np.clip(x, -c, c)
    s = 127.0 / c
    q = np.rint(x * s).astype(np.int8)
    return q

def dequantifier(q, c = 0.25):
    s = 127.0 / c
    x = q.astype(np.float32) / s
    return x

path = "Embedding/val"
folders = os.listdir(path)

for folder in folders:
    folderPath = f"{path}/{folder}"
    files = os.listdir(folderPath)

    for file in files:
        embFilePath = f"{folderPath}/{file}"

        with open(embFilePath, 'rb') as f:
            emb = pickle.load(f)
            embQ = quantifier(emb)
            embDQ = dequantifier(embQ)
            embDQ = embDQ / (np.linalg.norm(embDQ) + 1e-12)
            score = np.dot(emb, embDQ)
            print(score)
