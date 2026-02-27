import numpy as np
import zlib, cv2, os
from reedsolo import RSCodec, ReedSolomonError
from PIL import Image

def getIdentity(app, path):
    img = cv2.imread(path)
    faces = app.get(img)

    if(len(faces) == 0):
        return None

    faces = sorted(faces, key = lambda f: f.bbox[2] * f.bbox[3], reverse = True)
    emb = faces[0].normed_embedding

    return emb, faces[0].bbox

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

def createHeader(embQ):
    crc = zlib.crc32(embQ) & 0xffffffff
    header = b"ID" + crc.to_bytes(4, byteorder = "big")
    return header

def confirmerHeader(data: bytes):
    magic = data[:2]
    if(magic != b"ID"): return None

    crc = int.from_bytes(data[2:6], "big")
    payload = data[6:6 + 512]

    crcCheck = zlib.crc32(payload) & 0xffffffff
    if(crcCheck != crc): return None

    return True

def createECC(data, nsym):
    rsc = RSCodec(nsym)
    return rsc.encode(data)

def decodeECC(data, nsym):
    rsc = RSCodec(nsym)
    try:
        result = rsc.decode(data)[0]
    except ReedSolomonError:
        return None

    return result

def createAnchor(cornerID):
    anchor = np.zeros((7, 7), dtype = np.uint8)

    anchor[0, :] = 1
    anchor[6, :] = 1
    anchor[:, 0] = 1
    anchor[:, 6] = 1
    anchor[3, 3] = 1

    b0 = (cornerID >> 0) & 1
    b1 = (cornerID >> 1) & 1
    anchor[1, 5] = b0
    anchor[5, 1] = b1

    return anchor

def downloadQRimage(code, path):
    margin = 0
    img = ((1-code) * 255).astype(np.uint8)

    H, W = img.shape
    canvas = np.full((H + 2*margin, W + 2*margin), 255, dtype = np.uint8)
    canvas[margin : margin + H, margin : margin + W] = img

    canvas = cv2.resize(
        canvas,
        (canvas.shape[1]*10, canvas.shape[0]*10),
        interpolation=cv2.INTER_NEAREST
    )

    os.makedirs(os.path.dirname(path), exist_ok = True)
    Image.fromarray(canvas, mode = "L").save(path)

def expandBbox(bbox, img_h, img_w, margin = 0.1):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1

    dx, dy = int(w * margin), int(h * margin)
    bboxNew = [max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(img_w, int(x2 + dx)), min(img_h, int(y2 + dy))]

    return bboxNew

def usableBlock(hc, wc, bbox, block = 8):
    block_x = wc // block
    block_y = hc // block

    usable = np.ones((block_y, block_x), dtype = np.uint8)
    x1, y1, x2, y2 = bbox

    bx1, bx2 = x1 // block, (x2 - 1) // block
    by1, by2 = y1 // block, (y2 - 1) // block

    usable[by1:by2+1, bx1:bx2+1] = 0

    return np.ascontiguousarray(usable.reshape(-1), dtype = np.uint8)
