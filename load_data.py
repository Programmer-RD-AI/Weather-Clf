import cv2
import os
import torch
import numpy as np


def load_data(directory="./data/data/"):
    IMG_SIZE = 112
    LABELS = {}
    DATA = []
    index = -1
    for folder in os.listdir(directory):
        index += 1
        LABELS[f"./data/data/{folder}/"] = [index, -1]
    print(len(LABELS))
    for label in LABELS:
        for file in os.listdir(label):
            try:
                filepath = label + file
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                LABELS[label][1] += 1
                DATA.append([np.array(img), LABELS[label][0]])
            except Exception as e:
                print(e)
                print(filepath)
    np.random.shuffle(DATA)
    np.random.shuffle(DATA)
    
    np.save("./data/data.npy", DATA)
    VAL_SPLIT = 0.25
    X, y = [], []
    for d in DATA:
        X.append(d[0])
        y.append(d[1])
    VAL_SPLIT = len(X) * VAL_SPLIT
    VAL_SPLIT = int(VAL_SPLIT)
    X_train = X[:-VAL_SPLIT]
    y_train = y[:-VAL_SPLIT]
    X_test = X[-VAL_SPLIT:]
    y_test = y[-VAL_SPLIT:]
    X_train = torch.from_numpy(np.array(X_train))
    y_train = torch.from_numpy(np.array(y_train))
    X_test = torch.from_numpy(np.array(X_test))
    y_test = torch.from_numpy(np.array(y_test))
    return [X_train, X_test, y_train, y_test]
