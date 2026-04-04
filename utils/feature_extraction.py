"""
feature_extraction.py
---------------------
Handcrafted feature extraction pipeline for guava ripeness prediction.
Extracts: LAB stats, HSV stats, GLCM texture, LBP texture, shape features.
Total feature vector: ~200+ dims
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ── LAB Features ────────────────────────────────────────────────────────────

def extract_lab_features(img_bgr):
    """Mean, std per LAB channel + normalized histogram (32 bins each)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    feats = []

    for i in range(3):
        ch = lab[:, :, i].astype(np.float32)
        feats += [ch.mean(), ch.std()]                          # 2 per channel = 6

        hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
        hist = (hist / hist.sum()).flatten()
        feats.append(hist)                                      # 32 per channel = 96

    return np.concatenate([np.array(feats[:6])] +
                          [f if isinstance(f, np.ndarray) else np.array([f])
                           for f in feats[6:]])


def _lab_stats_and_hist(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    stats, hists = [], []
    for i in range(3):
        ch = lab[:, :, i].astype(np.float32)
        stats += [ch.mean(), ch.std()]
        h = cv2.calcHist([lab], [i], None, [32], [0, 256])
        hists.append((h / h.sum()).flatten())
    return np.array(stats), np.concatenate(hists)   # (6,) + (96,)


# ── HSV Features ────────────────────────────────────────────────────────────

def extract_hsv_features(img_bgr):
    """Mean and std for each HSV channel."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    feats = []
    for i in range(3):
        ch = hsv[:, :, i]
        feats += [ch.mean(), ch.std()]
    return np.array(feats)   # (6,)


# ── GLCM Texture Features ────────────────────────────────────────────────────

def extract_glcm_features(img_bgr):
    """
    GLCM at 4 angles × 2 distances.
    Properties: contrast, dissimilarity, homogeneity, energy, correlation, ASM.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    distances = [1, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props     = ['contrast', 'dissimilarity', 'homogeneity',
                 'energy', 'correlation', 'ASM']

    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    feats = []
    for p in props:
        vals = graycoprops(glcm, p)   # shape (len(distances), len(angles))
        feats += [vals.mean(), vals.std()]

    return np.array(feats)   # (12,)


# ── LBP Texture Features ─────────────────────────────────────────────────────

def extract_lbp_features(img_bgr, P=24, R=3, n_bins=64):
    """
    Uniform LBP histogram — captures micro-texture patterns
    highly correlated with surface ripeness.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, P + 2), density=True)
    return hist.astype(np.float32)   # (64,)


# ── Shape Features ───────────────────────────────────────────────────────────

def extract_shape_features(img_bgr):
    """
    Segment the fruit and compute area, perimeter, circularity,
    aspect ratio, extent, solidity.
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # broad green/yellow mask for guava
    mask = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([95, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(6, dtype=np.float32)

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    circularity = (4 * np.pi * area / (peri ** 2 + 1e-6))
    x, y, w, h  = cv2.boundingRect(cnt)
    aspect      = w / (h + 1e-6)
    extent      = area / (w * h + 1e-6)
    hull_area   = cv2.contourArea(cv2.convexHull(cnt))
    solidity    = area / (hull_area + 1e-6)

    total_px    = img_bgr.shape[0] * img_bgr.shape[1]
    return np.array([
        area / total_px,   # normalized area
        peri / (img_bgr.shape[0] + img_bgr.shape[1]),
        circularity,
        aspect,
        extent,
        solidity
    ], dtype=np.float32)   # (6,)


# ── Master Extractor ─────────────────────────────────────────────────────────

def extract_features(image_path: str) -> np.ndarray:
    """
    Full handcrafted feature vector.
    Dimensions:
        LAB stats      :   6
        LAB histograms :  96
        HSV stats      :   6
        GLCM           :  12
        LBP            :  64
        Shape          :   6
        ─────────────────────
        Total          : 190
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (224, 224))

    lab_stats, lab_hist = _lab_stats_and_hist(img)
    hsv   = extract_hsv_features(img)
    glcm  = extract_glcm_features(img)
    lbp   = extract_lbp_features(img)
    shape = extract_shape_features(img)

    return np.concatenate([lab_stats, lab_hist, hsv, glcm, lbp, shape]).astype(np.float32)


FEATURE_DIM = 190
