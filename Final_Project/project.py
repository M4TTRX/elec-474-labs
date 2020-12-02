import cv2
import numpy as np


def get_evalutaion_matrix(predictions, actual):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for p, a in zip(predictions, actual):
        if p == True and a == True:
            true_positive += 1
            continue
        if p == True and a == True:
            false_positive += 1
            continue
        if p == True and a == True:
            true_negative += 1
            continue
        if p == True and a == True:
            false_negative += 1
            continue
    return (
        true_positive,
        false_positive,
        true_negative,
        false_negative,
    )
