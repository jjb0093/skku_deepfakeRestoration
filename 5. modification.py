from module import getIdentity, downloadQRimage
from module import quantifier, dequantifier
from module import createHeader, confirmerHeader
from module import createECC, decodeECC
from module import createAnchor

import pickle, zlib, cv2
import numpy as np

from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def getCode(path):
    '''
    path = r"Data\train\n000002\0001_01.jpg"
    emb = getIdentity(app, path)
    '''
    #path = r"Embedding\train\n000002\0002_01.pkl"

    with open(path, 'rb') as f:
        emb = pickle.load(f)
    embQ = quantifier(emb).tobytes()

    header = createHeader(embQ)
    data = header + embQ
    data_ecc = createECC(data, 96)

    data_bits = np.unpackbits(
        np.frombuffer(data_ecc, dtype = np.uint8)
    )

    anchors = [createAnchor(i) for i in range(3)]

    code = np.zeros((86, 86), dtype = np.uint8)
    code[0:7, 0:7] = anchors[0]
    code[0:7, 86-7:86] = anchors[1]
    code[86-7:86, 86-7:86] = anchors[2]

    reserved = np.zeros_like(code, dtype = bool)
    reserved[0:7, 0:7] = True
    reserved[0:7, 86-7:86] = True
    reserved[86-7:86, 86-7:86] = True

    idx = 0
    for i in range(86):
        for k in range(86):
            if(reserved[i, k]): continue
            if(idx >= len(data_bits)): break
            code[i, k] = data_bits[idx]
            idx += 1

    if(idx != len(data_bits)): print("Length ERROR")

    QRfilePath = path.replace("Data", "QRimage")
    downloadQRimage(code, QRfilePath)
