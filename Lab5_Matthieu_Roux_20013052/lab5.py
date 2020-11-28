import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
from numpy.lib.shape_base import tile

lab_path = "Lab5_Matthieu_Roux_20013052/"
coaster_l_path = lab_path + "coaster_left.jpg"
coaster_r_path = lab_path + "coaster_right.jpg"

tape_l_path = lab_path + "tape_l.jpg"
tape_r_path = lab_path + "tape_r.jpg"

coaster_l_img = cv2.imread(coaster_l_path, 0)
coaster_r_img = cv2.imread(coaster_r_path, 0)

tape_l_img = cv2.imread(tape_l_path, 0)
tape_r_img = cv2.imread(tape_r_path, 0)


def resize_img(img, max_side=1000):
    """
    This function resizes an image, while keeping its aspect ratio,
    ensuring that its largest side is not greater
    than max_side (1000px by default).
    """
    height, width = img.shape
    # if the image is small enough as is, return it unchanged
    if height <= 1000 and width <= 1000:
        return img
    dim = tuple()
    if height > width:
        scale_ratio = max_side / height
        dim = (int(width * scale_ratio), max_side)
    else:
        scale_ratio = max_side / width
        dim = (max_side, int(height * scale_ratio))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def show_img(imgs, names=[], use_plt=False):
    # resize
    for i in range(len(imgs)):
        imgs[i] = resize_img(imgs[i], max_side=750)

    # create windows
    if len(names) == 0:
        names = [str(i) for i in range(len(imgs))]
    for name in names:
        cv2.namedWindow(name)

    while True:
        # Wait a little bit for the image to re-draw
        key = cv2.waitKey(5)
        for img, name in zip(imgs, names):
            cv2.imshow(name, img)

        # If an x is pressed, the window will close
        if key == ord("x"):
            break
    cv2.destroyAllWindows()


## Fundamental Matrix


def lowe_ratio_match(matches, threshold_ratio=0.5):
    lowe_matches = [m for m, n in matches if m.distance < threshold_ratio * n.distance]
    return lowe_matches


def compute_matches(des1, des2):
    bf = cv2.BFMatcher()
    return bf.knnMatch(des1, des2, k=2)


def get_matches(img_1, img_2):
    my_SIFT_instance = cv2.SIFT_create()
    # putting keypoints in variabels for better legibility
    kp1, des1 = my_SIFT_instance.detectAndCompute(img_1, None)
    kp2, des2 = my_SIFT_instance.detectAndCompute(img_2, None)

    # # matching
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # get lowe matches
    lowe_matches = lowe_ratio_match(matches, threshold_ratio=0.8)
    # get the index of points in matches
    matches_kp1 = np.int32([kp1[m.queryIdx].pt for m in lowe_matches])
    matches_kp2 = np.int32([kp2[m.trainIdx].pt for m in lowe_matches])
    return (
        matches_kp1,
        matches_kp2,
        lowe_matches,
    )


def get_fundamental_matrix(img1, img2):
    # get the matches
    pts_1, pts_2, _ = get_matches(img1, img2)

    fundamental_matrix, mask = cv2.findFundamentalMat(
        pts_1,
        pts_2,
        cv2.RANSAC,
        10,
    )
    pts_1 = pts_1[mask.ravel() == 1]
    pts_2 = pts_2[mask.ravel() == 1]
    return fundamental_matrix, (pts_1, pts_2)


def rectify_image(img1, img2, fundamental_matrix, inlier_matches):
    pts1, pts2 = inlier_matches
    _, homography1, homography2 = cv2.stereoRectifyUncalibrated(
        pts1, pts2, fundamental_matrix, img1.shape
    )
    new_img1 = cv2.warpPerspective(img1, homography1, img1.shape)
    new_img2 = cv2.warpPerspective(img1, homography2, img1.shape)
    return new_img1, new_img2


def get_disparity_img(img1, img2, numDisparities=16, blockSize=5):
    stereo = cv2.StereoBM_create(
        numDisparities=numDisparities,
        blockSize=blockSize,
    )
    img1 = cv2.GaussianBlur(src=img1, ksize=(0, 0), sigmaX=4)
    img2 = cv2.GaussianBlur(src=img2, ksize=(0, 0), sigmaX=4)
    disparity = stereo.compute(img1, img2)
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))
    disparity = resize_img(disparity, max_side=500)
    return disparity


f, inlier_matches = get_fundamental_matrix(coaster_l_img, coaster_r_img)
img1, img2 = rectify_image(coaster_l_img, coaster_r_img, f, inlier_matches)
show_img(imgs=[img1, img2], names=["Left Coaster", "Right Coaster"])
coaster_disparity = get_disparity_img(
    img1,
    img2,
    numDisparities=16,
    blockSize=13,
)


f, inlier_matches = get_fundamental_matrix(tape_l_img, tape_r_img)
img1, img2 = rectify_image(tape_l_img, tape_r_img, f, inlier_matches)
show_img(imgs=[img1, img2], names=["Left Tape", "Right Tape"])
tape_disparity = get_disparity_img(
    img1,
    img2,
    numDisparities=16,
    blockSize=7,
)

show_img(
    imgs=[coaster_disparity, tape_disparity],
    names=["Coaster Disparity", "Tape Disparity"],
)


# # Rendering
# window_name = "test"
# cv2.namedWindow(window_name)
# numDisparities = 16
# blockSize = 7
# # on_trackbar is called when the trackbar is changed, it updates the threshold
# def on_trackbar(val):
#     global k, coaster_disparity, numDisparities
#     numDisparities = val if val % 16 == 0 else val - val % 16
#     coaster_disparity = get_disparity_img(
#         img1,
#         img2,
#         numDisparities=numDisparities,
#         blockSize=blockSize,
#     )


# def on_trackbar2(val):
#     global k, coaster_disparity, blockSize
#     blockSize = val if val % 2 == 1 else val + 1
#     blockSize = 5 if blockSize < 5 else blockSize
#     coaster_disparity = get_disparity_img(
#         img1,
#         img2,
#         numDisparities=numDisparities,
#         blockSize=blockSize,
#     )


# # create trackbar
# cv2.createTrackbar("numDisparities", window_name, numDisparities, 256, on_trackbar)
# cv2.createTrackbar("blockSize", window_name, blockSize, 31, on_trackbar2)
# while True:
#     # Wait a little bit for the image to re-draw
#     key = cv2.waitKey(5)
#     cv2.imshow(window_name, coaster_disparity)

#     # If an x is pressed, the window will close
#     if key == ord("x"):
#         break