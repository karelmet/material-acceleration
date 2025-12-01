from __future__ import annotations

import numpy as np
import random 

# Fonctions de data augmentation simples

def random_horizontal_flip(img, boxes, p=0.5)-> tuple[np.ndarray, np.ndarray]:
    """
    Flip horizontal avec probabilité p.
    """
    if random.random() > p:
        return img, boxes
    
    img_filpped = img[:, ::-1, :]

    if boxes.size == 0:
        return img_filpped, boxes
    
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 1] = 1.0 - boxes_flipped[:, 1]

    return img_filpped, boxes_flipped

def random_vertical_flip(img, boxes, p=0.5)-> tuple[np.ndarray, np.ndarray]:
    """
    Flip vertical avec probabilité p.
    """
    if random.random() > p:
        return img, boxes

    img_flipped = img[::-1, :, :]

    if boxes.size == 0:
        return img_flipped, boxes

    boxes_flipped = boxes.copy()
    boxes_flipped[:, 2] = 1.0 - boxes_flipped[:, 2]

    return img_flipped, boxes_flipped

def random_brightness(img, boxes, p=0.5, max_delta=0.2)-> tuple[np.ndarray, np.ndarray]:
    """
    Change luminosité avec probabilité p.
    On travaille en float dans [0, 1] :
        img' = img + delta
    avec delta dans [-max_delta, +max_delta].
    On clip ensuite dans [0,1] et on revient en uint8.
    """
    if random.random() > p:
        return img, boxes
    
    delta = random.uniform(-max_delta, max_delta)
    img_f = img.astype(np.float32) / 255.0
    
    img_f = img_f + delta
    img_f = np.clip(img_f, 0, 1)

    img_out = (img_f * 255.0).astype(np.uint8)

    return img_out, boxes

def random_contrast(img, boxes, p=0.5, lower=0.3, upper=2.5)-> tuple[np.ndarray, np.ndarray]:
    """
    Change le contraste avec probabilité p et intensité < max_delta. 
    On travaille en float dans [0,1] :
        img' = mean + alpha * (img - mean)
    où alpha est dans [lower, upper].
    """
    if random.random() > p:
        return img, boxes

    alpha = random.uniform(lower, upper)
    img_f = img.astype(np.float32) / 255.0

    mean = img_f.mean(axis=(0, 1), keepdims=True)
    
    img_f = mean + alpha * (img_f - mean)
    img_f = np.clip(img_f, 0, 1)

    img_out = (img_f * 255.0).astype(np.uint8)

    return img_out, boxes

def simple_augmentation(img, boxes)-> tuple[np.ndarray, np.ndarray]:
    """
    Augmentation en utilisant les fonctions simples (flip, contraste et luminosité).
    """

    img, boxes = random_horizontal_flip(img, boxes)
    img, boxes = random_vertical_flip(img, boxes)
    img, boxes = random_brightness(img, boxes)
    img, boxes = random_contrast(img, boxes)

    return img, boxes


# À faire ensuite : fonctions d'augmentation plus compliquées