import pickle, cv2
import numpy as np
import ctypes as ct
import jpegio as jio
from pathlib import Path

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

    #bx1, bx2 = max(0, bx1), min(block_x-1, bx2)
    #by1, by2 = max(0, by1), min(block_y-1, by2)

    '''
    usable[by1 : by2 + 1, bx1 : bx2 + 1] = False
    usableBlock = [(by, bx) for by in range(block_y) for bx in range(block_x) if usable[by, bx]]
    '''
    usable[by1:by2+1, bx1:bx2+1] = 0

    return np.ascontiguousarray(usable.reshape(-1), dtype = np.uint8)

if(__name__ == "__main__"):
    with open("testData/testQRCode.pkl", 'rb') as f:
        code = pickle.load(f)
        codeShape = len(code)
        codeBit = code.reshape(-1).astype(np.uint8)
    with open("testData/testBbox.pkl", 'rb') as f:
        bbox = pickle.load(f)
    
    imgInputPath = "testData/testImage.jpg"
    imgOutputPath = imgInputPath.split('.')[0] + "_stegano." + imgInputPath.split('.')[-1]

    #img = cv2.imread(imgInputPath)
    #h, w = img.shape[:2]

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

    allumerF = True
    allumerS = True

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
