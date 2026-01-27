import os, pickle
import torch
import torch.nn.functional as F

path = "Embedding/train"
folders = os.listdir(path)

def same():
    final = []
    for folder in folders:
        folderPath = f"{path}/{folder}"
        files = os.listdir(folderPath)

        print(f"{folderPath}에 대한 유사도 측정", end = '')
        sim = []

        for i in range(len(files)):
            with open(f"{folderPath}/{files[i]}", 'rb') as f:
                stFile = pickle.load(f)
            id1 = torch.tensor(stFile)
            id1 = F.normalize(id1, dim = 0)

            for k in range(i + 1, len(files)):
                with open(f"{folderPath}/{files[k]}", 'rb') as f:
                    cmFile = pickle.load(f)
                id2 = torch.tensor(cmFile)
                id2 = F.normalize(id2, dim = 0)

                cosSim = torch.dot(id1, id2)
                sim.append(cosSim.item())
                
        mean = sum(sim) / len(sim)
        print(f"결과 : {mean}")
        final.append(mean)

    print(f"최종결과 : {sum(final) / len(final)}")

def diff():
    print("서로 다른 이미지들의 1번 파일에 대한 유사도 측정")
    sim = []

    for i in range(len(folders)):
        stFiles = sorted(os.listdir(f"{path}/{folders[i]}"))
        with open(f"{path}/{folders[i]}/{stFiles[0]}", 'rb') as f:
            stFile = pickle.load(f)
        id1 = torch.tensor(stFile)
        id1 = F.normalize(id1, dim = 0)

        for k in range(i + 1, len(folders)):
            cmFiles = sorted(os.listdir(f"{path}/{folders[k]}"))
            with open(f"{path}/{folders[k]}/{cmFiles[0]}", 'rb') as f:
                cmFile = pickle.load(f)
            id2 = torch.tensor(cmFile)
            id2 = F.normalize(id2, dim = 0)

            cosSim = torch.dot(id1, id2)
            sim.append(cosSim.item())

    print(f"결과 : {sum(sim) / len(sim)}")

same()
diff()