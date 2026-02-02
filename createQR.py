import os, pickle
from framework import getCode
path = "Embedding/train"
folders = os.listdir(path)

for folder in folders:
    folderPath = f"{path}/{folder}"
    files = os.listdir(folderPath)

    for file in files:
        embFilePath = f"{folderPath}/{file}"
        getCode(embFilePath)