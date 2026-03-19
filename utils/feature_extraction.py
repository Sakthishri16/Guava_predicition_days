import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_lab(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    features = []
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
        hist = hist / hist.sum()
        features.append(hist.flatten())

    return np.concatenate(features)


def extract_glcm(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ])


def extract_features(image_path):
    lab = extract_lab(image_path)
    glcm = extract_glcm(image_path)
    return np.concatenate([lab, glcm])