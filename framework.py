from module import getIdentity, downloadQRimage
from module import quantifier, dequantifier
from module import createHeader, confirmerHeader
from module import createECC, decodeECC
from module import createAnchor
from module import expandBbox, usableBlock

import pickle
import numpy as np
import ctypes as ct
import jpegio as jio
from pathlib import Path

from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def getCode(path):
    emb, bbox = getIdentity(app, path)
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

    QRfilePath = "testData/testQRimage.png"
    downloadQRimage(code, QRfilePath)
    
    #with open("testData/testQRCode.pkl", 'wb') as f:
    #    pickle.dump(code, f)

    return code, bbox

def steganoInfo(code, bbox, path):
    codeShape = len(code)
    codeBit = code.reshape(-1).astype(np.uint8)
    
    imgInputPath = path
    imgOutputPath = imgInputPath.split('.')[0] + "_stegano." + imgInputPath.split('.')[-1]

    imgJpeg = jio.read(imgInputPath)
    coef = imgJpeg.coef_arrays[0]
    h, w = imgJpeg.image_height, imgJpeg.image_width
    hc, wc = coef.shape

    bbox = expandBbox(bbox, h, w, 0.2)
    usable = usableBlock(hc, wc, bbox)

    margin = [max(0, (bbox[0]//8) - (codeShape//2)), max(0, ((bbox[3]-bbox[1])//2//8) - (codeShape//2))]

    print("coef.shape =", (hc, wc), "blocks =", (hc//8, wc//8), flush=True)
    print("usable.len =", len(usable), "usable.sum =", int(usable.sum()), flush=True)
    print("bits.len   =", len(codeBit), flush=True)

    dll_path = Path(__file__).resolve().parent / "infoInsertion.dll"
    dll = ct.CDLL(str(dll_path))

    uvPair = ct.c_int * 2

    dll.embedding.argtypes = [
        ct.c_char_p, ct.c_char_p,
        ct.POINTER(ct.c_ubyte), ct.c_size_t,
        ct.POINTER(ct.c_ubyte),
        ct.c_int, ct.c_int,
        ct.POINTER(uvPair), ct.c_int, ct.c_int,
        ct.c_int, ct.c_int
    ]
    dll.embedding.restype = ct.c_int

    uvList = (uvPair * 2)(uvPair(2, 3), uvPair(3, 2))
    uvCount = ct.c_int(2)

    bitsPtr = codeBit.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    usablePtr = usable.ctypes.data_as(ct.POINTER(ct.c_ubyte))

    allumerF = False
    allumerS = False

    result = dll.embedding(
        imgInputPath.encode("utf-8"),
        imgOutputPath.encode("utf-8"),
        bitsPtr, ct.c_size_t(len(codeBit)),
        usablePtr,
        ct.c_int(margin[0]), ct.c_int(margin[1]),
        uvList, uvCount,
        ct.c_int(codeShape), ct.c_int(1 if(allumerF) else 0), ct.c_int(1 if(allumerS) else 0),
        ct.c_int(bbox[0]), ct.c_int(bbox[1]), ct.c_int(bbox[2]), ct.c_int(bbox[3]), 
    )

    print("Embedding Result =", result)

if(__name__ == "__main__"):
    path = "testData/testImage.jpg"
    code, bbox = getCode(path)
    steganoInfo(code, bbox, path)

