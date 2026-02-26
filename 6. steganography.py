import pickle, cv2
import numpy as np
import jpegio as jio

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

def setParity(c, bit):
    if(c == 0): return 0 if bit == 0 else 1
    if(c&1 == bit): return c
    return c + (1 if c>0 else -1)

def embedding(image, code, usable, unusable, uvList = [(2, 3), (3, 2)]):
    imgJpeg = jio.read(image)
    coef = imgJpeg.coef_arrays[0]
    length = len(code) * len(code[0]) // len(uvList)

    h, w = coef.shape
    print(f"가로 블럭 {w//8}개, 세로 블럭 {h//8}개, 총 {w*h//64}개")
    print(f"사용해야 하는 블럭 수 : {length}")
    print(f"사용가능 블럭 수 : {w*h//64 - unusable}")
    
    #blockNum = [0, 0]
    #block = coef[8*blockNum[0] : 8*(blockNum[0]+1), 8*blockNum[1] : 8*(blockNum[1]+1)]

    # 1. 특정 좌표부터 얼굴 부분을 제외하여 정사각형 그리기
    margin = [50, 40]
    usableCount, rowCount = 0, 0
    for i in range(len(code)):
        for j in range(w//8):
            if((rowCount + margin[0], j + margin[1]) in usable):
                block = coef[8*(i+margin[0]) : 8*(i+margin[0]+1), 8*(j+margin[1]) : 8*(j+margin[1]+1)]
                block[0, 0] = int(block[0, 0]) + 200
                #block[2, 3] = setParity(block[2, 3], code[usableCount // len(code[0])][usableCount % len(code[0])])
                block[3, 2] = setParity(block[3, 2], code[usableCount // len(code[0])][usableCount % len(code[0])])
                #for(u, v) in uvList:
                    #block[u, v] = setParity(block[u][v], code[usableCount // len(code[0])][usableCount % len(code[0])])]
                    #block[u, v] = 5
                usableCount += 1

            if(usableCount % len(code[0]) == 0):
                rowCount += 1
                break

    path = "testData/testImageStegano.jpg"
    jio.write(imgJpeg, path)

if(__name__ == "__main__"):
    with open("testData/testQRCode.pkl", 'rb') as f:
        code = pickle.load(f)
    with open("testData/testBbox.pkl", 'rb') as f:
        bbox = pickle.load(f)
    
    imgPath = "testData/testImage.jpg"
    img = cv2.imread(imgPath)
    h, w = img.shape[:2]

    bbox = expandBbox(bbox, h, w)
    usable = usableBlock(h, w, bbox)
    unusable = ((bbox[2]-bbox[0])//8) * ((bbox[3]-bbox[1])//8)

    embedding(imgPath, code, usable, unusable)
    
