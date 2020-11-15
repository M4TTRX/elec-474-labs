import numpy as np
import cv2
from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from copy import deepcopy


prelab_path = "Lab4_Matthieu_Roux_20013052/"
# Reference image import
cereal_url = prelab_path + "cereal.jpg"

img_ref_bgr = cv2.imread(cereal_url)
img_ref = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2GRAY)

# Perspective import
cereal_r_url = prelab_path + "cereal_r.jpg"
cereal_l_url = prelab_path + "cereal_l.jpg"
cereal_tr_url = prelab_path + "cereal_tr.jpg"
cereal_tl_url = prelab_path + "cereal_tl.jpg"
cereal_per_url = cereal_tl_url

cereal_r = cv2.cvtColor(cv2.imread(cereal_r_url), cv2.COLOR_BGR2GRAY)
cereal_l = cv2.cvtColor(cv2.imread(cereal_l_url), cv2.COLOR_BGR2GRAY)
cereal_tr = cv2.cvtColor(cv2.imread(cereal_tr_url), cv2.COLOR_BGR2GRAY)
cereal_tl = cv2.cvtColor(cv2.imread(cereal_tl_url), cv2.COLOR_BGR2GRAY)


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


def lowe_ratio_match(matches, threshold_ratio=0.7):
    return [m for m, n in matches if m.distance < threshold_ratio * n.distance]


def compute_matches(des1, des2):
    bf = cv2.BFMatcher()
    return bf.knnMatch(des1, des2, k=2)


def get_matches(img_1, img_2):
    my_SIFT_instance = cv2.SIFT_create()
    # putting keypoints in variabels for better legibility
    kp1, des1 = my_SIFT_instance.detectAndCompute(img_1, None)
    kp2, des2 = my_SIFT_instance.detectAndCompute(img_2, None)

    # matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    lowe_matches = lowe_ratio_match(matches)
    return (kp1, des1), (kp2, des2), lowe_matches


def affine_transform(img, rotation_value=42, scaling_value=0.4):
    height, width = img.shape
    center_index = (int(height / 2), int(width / 2))
    rotation_matrix = cv2.getRotationMatrix2D(
        center_index, rotation_value, scaling_value
    )
    new_img = cv2.warpAffine(img, rotation_matrix, (height, width))
    return new_img


def make_red(img):
    img_red = deepcopy(img)
    for index in np.ndindex(img_red.shape[:2]):
        # if an image is too blue and not red enough...
        if img_red[index][0] > img_red[index][2] * 1.25:
            # ...swap the red and blue values!
            temp = img_red[index][0]
            img_red[index][0] = img_red[index][2]
            img_red[index][2] = temp
    return img_red


def is_black(pixel):
    for sub_px in pixel:
        if sub_px > 0:
            return False
    return True


def apply_overlay(background, overlay):
    img_overlaid = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for index in np.ndindex(img_overlaid.shape[:2]):
        try:
            if not is_black(overlay[index]):
                img_overlaid[index] = overlay[index]
        except:
            pass
    return img_overlaid


# generate an affine image
img_affine = affine_transform(img_ref)

# obtain matches and keypoints
ref_params, affine_params, lowe_matches = get_matches(img_ref, img_affine)

kp_ref, des_ref = ref_params
kp_affine, des_affine = affine_params

# format points
ref_pts = np.float32(
    [kp_ref[m.queryIdx].pt for m in lowe_matches],
).reshape(-1, 1, 2)
img_pts = np.float32(
    [kp_affine[m.trainIdx].pt for m in lowe_matches],
).reshape(-1, 1, 2)

estimated_rotation_matrix = cv2.estimateAffinePartial2D(ref_pts, img_pts)[0]

modified_img = make_red(img_ref_bgr)
affine_overlay = cv2.warpAffine(
    modified_img,
    estimated_rotation_matrix,
    (modified_img.shape[0], modified_img.shape[1]),
)
img_affine_overlaid = apply_overlay(background=img_affine, overlay=affine_overlay)

# Perspective
def perspective_overlay(
    img_ref, img_perspective, modified_img, display_img=True, name="my perspectie test"
):
    # obtain matches and keypoints
    ref_params, affine_params, lowe_matches = get_matches(img_ref, img_perspective)

    kp_ref = ref_params[0]
    kp_perspective = affine_params[0]

    # format points
    ref_pts = np.float32(
        [kp_ref[m.queryIdx].pt for m in lowe_matches],
    ).reshape(-1, 1, 2)
    img_pts = np.float32(
        [kp_perspective[m.trainIdx].pt for m in lowe_matches],
    ).reshape(-1, 1, 2)

    homography_matrix = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC)[0]

    img_overlay = cv2.warpPerspective(
        modified_img,
        homography_matrix,
        (modified_img.shape[0], modified_img.shape[1]),
    )
    img_per_overlaid = apply_overlay(background=img_perspective, overlay=img_overlay)

    if display_img:
        show_img(img_per_overlaid, name="my perspectie test")

    return img_per_overlaid


# Displaying
show_img(cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR), name="reference")

# Display affine images, original then overlay
show_img(cv2.cvtColor(img_affine, cv2.COLOR_GRAY2BGR), name="affine_original")
show_img(img_affine_overlaid, name="affine_overlaid")

# Display perspective images, original then overlay
show_img(cv2.cvtColor(cereal_r, cv2.COLOR_GRAY2BGR), name="cereal_r_original")
perspective_overlay(img_ref, cereal_r, modified_img, name="cereal_r_overlaid")

show_img(cv2.cvtColor(cereal_l, cv2.COLOR_GRAY2BGR), name="cereal_l_original")
perspective_overlay(img_ref, cereal_l, modified_img, name="cereal_l_overlaid")

show_img(cv2.cvtColor(cereal_tr, cv2.COLOR_GRAY2BGR), name="cereal_tr_original")
perspective_overlay(img_ref, cereal_tr, modified_img, name="cereal_tr_overlaid")

show_img(cv2.cvtColor(cereal_tl, cv2.COLOR_GRAY2BGR), name="cereal_tl_original")
perspective_overlay(img_ref, cereal_tl, modified_img, name="cereal_tl_overlaid")

# Show all images