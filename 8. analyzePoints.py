import numpy as np
import pickle as pkl

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

anchors = [createAnchor(i) for i in range(3)]
