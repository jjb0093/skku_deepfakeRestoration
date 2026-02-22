import pickle, cv2
import numpy as np

def expandBbox(bbox, img_h, img_w, margin = 0.1):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1

    dx, dy = int(w * margin), int(h * margin)
    bboxNew = [max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(img_w, int(x2 + dx)), min(img_h, int(y2 + dy))]

    return bboxNew

def usableBlock(img_h, img_w, bbox, block = 8):
    block_x = img_w // block
    block_y = img_h // block

    usable = np.ones((block_y, block_x), dtype = bool)
    x1, y1, x2, y2 = bbox

    bx1, bx2 = x1 // block, (x2 - 1) // block
    by1, by2 = y1 // block, (y2 - 1) // block

    usable[by1 : by2 + 1, bx1 : bx2 + 1] = False
    usableBlock = [(by, bx) for by in range(block_y) for bx in range(block_x) if usable[by, bx]]

    return usableBlock

def embeding(image, code, usable, uvList = [(2, 3), (3, 2)]):
    h, w = image.shape[:2]

if(__name__ == "__main__"):
    with open("testData/testQRCode.pkl", 'rb') as f:
        code = pickle.load(f)
    with open("testData/testBbox.pkl", 'rb') as f:
        bbox = pickle.load(f)
    img = cv2.imread("testData/testImage.jpg")
    h, w = img.shape[:2]

    bbox = expandBbox(bbox, h, w)
    usable = usableBlock(h, w, bbox)
    
    print(f"QR SIZE : {len(code) * len(code[-1])}")
    print(f"Usable Blocks : {len(usable)}")