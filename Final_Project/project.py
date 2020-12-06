import cv2
import numpy as np
from lib.haar_cascade import FaceFinder
from lib.helpers import show_img


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


def import_dataset():
    # initiate img types and path
    img_types = ["C", "L", "R"]
    data_path = "Final_Project/dataset/"

    real_imgs = [
        [cv2.imread(f"{data_path}real/{i}{t}r.jpg") for t in img_types]
        for i in range(1, 12 + 1)
    ]
    fake_imgs = [
        [cv2.imread(f"{data_path}fake/{i}{t}f.jpg") for t in img_types]
        for i in range(1, 12 + 1)
    ]
    return real_imgs, fake_imgs


if __name__ == "__main__":

    # load the dataset
    real_imgs, fake_imgs = import_dataset()

    # find the faces in all images
    path = "../../opencv"
    face_finder = FaceFinder(path)
    real_faces = [
        [face_finder.find_face(img) for img in triplet] for triplet in real_imgs
    ]
    for triplet in real_faces:
        show_img(triplet)
    fake_faces = [
        [face_finder.find_face(img) for img in triplet] for triplet in fake_imgs
    ]
