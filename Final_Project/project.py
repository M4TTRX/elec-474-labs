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


def import_images():
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
    test_imgs = [
        [cv2.imread(f"{data_path}test/{i}{t}t.jpg") for t in img_types]
        for i in range(1, 6 + 1)
    ]
    return real_imgs, fake_imgs, test_imgs


def matches(img1, img2):
    my_SIFT_instance = cv2.SIFT_create()
    kp1, des1 = my_SIFT_instance.detectAndCompute(img1, None)
    kp2, des2 = my_SIFT_instance.detectAndCompute(img2, None)
    lowes = cv2.FlannBasedMatcher().knnMatch(
        np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2
    )

    leftPoints, rightPoints = [], []
    for m, n in lowes:
        if (m.distance / n.distance) < 0.8:
            rightPoints.append(kp2[m.trainIdx].pt)
            leftPoints.append(kp1[m.queryIdx].pt)

    leftPoints, rightPoints = np.int32(leftPoints), np.int32(rightPoints)

    return len(leftPoints)


def get_triplet_match(triplet, show_imgs=False):
    if show_imgs:
        show_img(triplet, names=["C", "L", "R"])

    center, left, right = triplet
    if center is None:
        return matches(left, right)
    if left is None:
        return matches(center, right)
    if right is None:
        return matches(center, left)

    # by now they should all exist
    # remove the smallest image in the triplet as there is sometimes a small image that is not a face
    sizes = [(img.shape[0] * img.shape[1], img) for img in triplet]
    try:
        sizes.remove(min(sizes))
    except:
        sizes = [sizes[0], sizes[1]]

    return matches(sizes[0][1], sizes[1][1])


def train(real_imgs, fake_imgs, face_finder):
    real_faces = [face_finder.detect_triplet(triplet) for triplet in real_imgs]
    fake_faces = [face_finder.detect_triplet(triplet) for triplet in fake_imgs]

    real_face_matches = [
        get_triplet_match(real_face_triplet) for real_face_triplet in real_faces
    ]
    fake_face_matches = [
        get_triplet_match(fake_face_triplet) for fake_face_triplet in fake_faces
    ]

    match_threshold = (max(real_face_matches) + min(fake_face_matches)) // 2

    return match_threshold


def detect(triplet, threshold):
    matches = get_triplet_match(triplet, show_imgs=True)
    if matches < threshold:
        return True
    return False


if __name__ == "__main__":
    # load the dataset
    real_imgs, fake_imgs, test_imgs = import_images()

    path = "Final_Project/res"
    face_finder = FaceFinder(path)

    match_threshold = train(real_imgs, fake_imgs, face_finder)

    test_faces = [face_finder.detect_triplet(triplet) for triplet in test_imgs]
    actual = [True, False, True, True, True, True]

    results = [detect(triplet, match_threshold) for triplet in test_faces]

    print(actual)
    print(results)