import numpy as np
import cv2
from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt


prelab_path = "Lab4_Matthieu_Roux_20013052/"
cereal_url = prelab_path + "cereal.jpg"
cereal_r_url = prelab_path + "cereal_r.jpg"

img_ref_bgr = cv2.imread(cereal_url)
img_ref = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2GRAY)
img_perspective = cv2.cvtColor(cv2.imread(cereal_r_url), cv2.COLOR_BGR2GRAY)


def show_img(img, name="my image"):
    cv2.namedWindow(name)
    while True:
        # Wait a little bit for the image to re-draw
        key = cv2.waitKey(5)
        cv2.imshow(name, img)

        # If an x is pressed, the window will close
        if key == ord("x"):
            break
    cv2.destroyAllWindows()


def lowe_ratio_match(matches, threshold_ratio=0.5):
    return [m for m, n in matches if m.distance < threshold_ratio * n.distance]


def compute_matches(des1, des2):
    bf = cv2.BFMatcher()
    return bf.knnMatch(des1, des2, k=2)


def get_matches(img_1, img_2):
    my_SIFT_instance = cv2.SIFT_create()
    # putting keypoints in variabels for better legibility
    kp_img_1, des_img_1 = my_SIFT_instance.detectAndCompute(img_1, None)
    kp_img_2, des_img_2 = my_SIFT_instance.detectAndCompute(img_2, None)

    matches = compute_matches(des_img_1, des_img_2)
    lowe_ratio_matches = lowe_ratio_match(matches)
    return (kp_img_1, des_img_1), (kp_img_2, des_img_2), lowe_ratio_matches


def affine_transform(img, rotation_value=42, scaling_value=0.4):
    height, width = img.shape
    center_index = (int(height / 2), int(width / 2))
    rotation_matrix = cv2.getRotationMatrix2D(
        center_index, rotation_value, scaling_value
    )
    new_img = cv2.warpAffine(img, rotation_matrix, (height, width))
    return new_img


def modify_image(img):
    # this is a quick and easy way to change the colours in an interesting way
    return img * 2


def is_black(pixel):
    for sub_px in pixel:
        if sub_px > 0:
            return False
    return True


def apply_overlay(background, overlay):
    img_overlaid = np.zeros((background.shape[0], background.shape[1], 3), np.uint8)
    for index in np.ndindex(background.shape):
        img_overlaid[index] = (
            background[index] if is_black(overlay[index]) else overlay[index]
        )
    return img_overlaid


# generate an affine image
img_affine = affine_transform(img_ref)

# obtain matches and keypoints
ref_params, affine_params, lowe_matches = get_matches(img_ref, img_affine)

kp_ref, des_ref = ref_params
kp_affine, affine_des = affine_params

# format points
ref_pts = np.float32(
    [kp_ref[m.queryIdx].pt for m in lowe_matches],
).reshape(-1, 1, 2)
img_pts = np.float32(
    [kp_affine[m.trainIdx].pt for m in lowe_matches],
).reshape(-1, 1, 2)

estimated_rotation_matrix = cv2.estimateAffinePartial2D(ref_pts, img_pts)[0]

modified_img = modify_image(img_ref_bgr)
overlaid_affine = cv2.warpAffine(
    modified_img,
    estimated_rotation_matrix,
    (modified_img.shape[0], modified_img.shape[1]),
)

# Perspective

# obtain matches and keypoints
ref_params, affine_params, lowe_matches = get_matches(img_ref, img_perspective)

kp_ref, des_ref = ref_params
kp_affine, affine_des = affine_params

# format points
ref_pts = np.float32(
    [kp_ref[m.queryIdx].pt for m in lowe_matches],
).reshape(-1, 1, 2)
img_pts = np.float32(
    [kp_affine[m.trainIdx].pt for m in lowe_matches],
).reshape(-1, 1, 2)

homography_matrix = cv2.findHomography(ref_pts, img_pts)[0]
modified_img = modify_image(img_ref_bgr)

img_overlay = cv2.warpPerspective(
    modified_img,
    homography_matrix,
    (modified_img.shape[0], modified_img.shape[1]),
)
show_img(apply_overlay(background=img_perspective, overlay=img_overlay))