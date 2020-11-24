import cv2
import numpy as np

# imports

prelab_path = "Prelab5_Matthieu_Rou_20013052/"
cmpe_l_path = prelab_path + "computers_left.png"
cmpe_r_path = prelab_path + "computers_right.png"

daft_punk_l_path = prelab_path + "daft_punk_left.jpg"
daft_punk_r_path = prelab_path + "daft_punk_right.jpg"

cmpe_l_img = cv2.cvtColor(cv2.imread(cmpe_l_path), cv2.COLOR_BGR2GRAY)
cmpe_r_img = cv2.cvtColor(cv2.imread(cmpe_r_path), cv2.COLOR_BGR2GRAY)
daft_punk_l_img = cv2.cvtColor(cv2.imread(daft_punk_l_path), cv2.COLOR_BGR2GRAY)
daft_punk_r_img = cv2.cvtColor(cv2.imread(daft_punk_r_path), cv2.COLOR_BGR2GRAY)

keypoints = "keypoints"
descriptors = "descriptors"
match_descriptors = "match_descriptors"


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


# match finding


def lowe_ratio_match(matches, threshold_ratio=0.7):
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

    # matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

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


def line_maker(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    row_num, column_num = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    lines = lines.reshape(-1, 3)
    for line, pt1 in zip(lines, pts1):
        # Finding a random color !
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # find left bound point
        left_bound = map(int, [0, -line[2] / line[1]])

        # find right bound point
        right_bound = map(
            int, [column_num, -(line[2] + line[0] * column_num) / line[1]]
        )

        # draw the line on the image
        img1 = cv2.line(img1, tuple(left_bound), tuple(right_bound), color, 1)

        # draw the point on image 1
        img1 = cv2.circle(img1, tuple(pt1[0]), 5, color, -1)
    return img1


def epipolar_line_calculation(img1, img2):
    # get the matches
    pts_1, pts_2, _ = get_matches(img1, img2)

    fundamental_matrix, mask = cv2.findFundamentalMat(
        pts_1,
        pts_2,
        cv2.FM_LMEDS,
    )

    pts_1 = pts_1[mask.ravel() == 1]
    pts_2 = pts_2[mask.ravel() == 1]

    pts_1 = pts_1.reshape(-1, 1, 2)
    pts_2 = pts_2.reshape(-1, 1, 2)

    lines_1 = cv2.computeCorrespondEpilines(pts_2, 2, fundamental_matrix)
    lines_1 = lines_1.reshape(-1, 3)
    lines_2 = cv2.computeCorrespondEpilines(pts_1, 1, fundamental_matrix)
    lines_2 = lines_2.reshape(-1, 3)

    img1_with_lines = line_maker(
        img2,
        img1,
        lines_2,
        pts_2,
        pts_1,
    )
    img2_with_lines = line_maker(
        img1,
        img2,
        lines_1,
        pts_1,
        pts_2,
    )

    combined_img = cv2.hconcat([img1_with_lines, img2_with_lines])

    show_img(combined_img)


epipolar_line_calculation(cmpe_l_img, cmpe_r_img)
epipolar_line_calculation(daft_punk_l_img, daft_punk_r_img)
img_arr = [
    (daft_punk_l_img, daft_punk_r_img),  # click on left watch on right
    (daft_punk_r_img, daft_punk_l_img),  # click on right watch on left
    (cmpe_l_img, cmpe_r_img),  # click on left watch on right
    (cmpe_r_img, cmpe_l_img),  # click on right watch on left
]


def onMouse(event, _x, _y, flags, param):
    global img1, img2
    if event == cv2.EVENT_LBUTTONDOWN:
        # extract our point and draw it
        pt = np.asarray([_x, _y])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.circle(img1, (_x, _y), 5, color, -1)

        # get fundamental matrix
        pts_1, pts_2, _ = get_matches(img1, img2)

        fundamental_matrix, mask = cv2.findFundamentalMat(
            pts_1,
            pts_2,
            cv2.FM_LMEDS,
        )

        # get and draw the epiline
        line = cv2.computeCorrespondEpilines(
            pt.reshape(-1, 1, 2), 2, fundamental_matrix
        )
        line = line.reshape(-1)
        _, c, _ = img2.shape
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)

    return


window_name_1 = "img1"
window_name_2 = "img2"

cv2.namedWindow(window_name_1)
cv2.namedWindow(window_name_2)

img_arr = [(cmpe_l_img, cmpe_r_img), (daft_punk_l_img, daft_punk_r_img)]

for img_pair in img_arr:
    img1, img2 = img_pair

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    while True:
        # Detect mouse clicks
        cv2.setMouseCallback(window_name_1, onMouse)

        # Wait a little bit for the image to re-draw
        key = cv2.waitKey(1)
        cv2.imshow(window_name_1, img1)
        cv2.imshow(window_name_2, img2)

        # If an x is pressed, the window will close
        if key == ord("x"):
            break
    cv2.destroyAllWindows()