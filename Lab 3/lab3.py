from math import sqrt
import numpy as np
import cv2
from numpy.core.fromnumeric import size

# terminal colours
TGREEN = "\033[32m"  # Green character
TYELLOW = "\033[93m"  # Yellow text character
ENDC = "\033[m"  # reset to the defaults


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill=TYELLOW + ".",
    printEnd="",
):
    """
    Call in a loop to create terminal progress bar.
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)d
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + " " * (length - filledLength)
    print(f"\r{ENDC}{prefix} |{bar} {ENDC}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


img = cv2.cvtColor(cv2.imread("Lab 3/concentric_circles.jpg"), cv2.COLOR_BGR2GRAY)

threshold_low = 50
threshold_hi = 150

# use edge detection on img
img_canny = cv2.Canny(img, threshold1=threshold_low, threshold2=threshold_hi)


def get_points():
    return np.argwhere(img_canny == 255)


## 1.2 Point Circle Fitting
def get_center(points, img_shape, verbose=False):
    # extract points in (x,y) coordinates
    point1 = (points[0][1], points[0][0])
    point2 = (points[1][1], points[1][0])
    point3 = (points[2][1], points[2][0])

    vector1 = (point2[0] - point1[0], point2[1] - point1[1])
    vector2 = (point3[0] - point1[0], point3[1] - point1[1])

    # Use the vectors to calculate the mid points between the point1 -> point 2
    # and point1 -> point3
    mid_1 = (point1[0] + vector1[0] / 2, point1[1] + vector1[1] / 2)
    mid_2 = (point1[0] + vector2[0] / 2, point1[1] + vector2[1] / 2)

    # Calculate line slopes m_1 and m_2
    m_1 = vector1[1] / vector1[0]
    m_2 = vector2[1] / vector2[0]

    # co-linearity check
    if m_2 == m_1:
        if verbose:
            print("points are colinear")
        return None

    if m_2 == 0 or m_1 == 0:
        if verbose:
            print("Ignoringhorizontal slopes")
        return None
    # Find perpendicular slopes m_per_1 and m_per_2
    m_per_1 = (
        -1 / m_1 if m_1 != 0 else 10 ** 10
    )  # using a very large number works better than using infinity
    m_per_2 = (
        -1 / m_2 if m_2 != 0 else 10 ** 10
    )  # using a very large number works better than using infinity

    # Calculate the x of center of the circle
    x_c = (mid_2[1] - mid_2[0] * m_per_2 - (mid_1[1] - mid_1[0] * m_per_1)) / (
        m_per_1 - m_per_2
    )

    # Now that we have x we can find y
    y_c = x_c * m_per_1 + (mid_1[1] - mid_1[0] * m_per_1)

    if y_c > img_shape[0] or x_c > img_shape[1]:
        if verbose:
            print("Center of the circle is outside the bounds of the image")
        return None
    ## Calculate the radius
    r = sqrt((x_c - point1[0]) ** 2 + (y_c - point1[1]) ** 2)
    return (int(x_c), int(y_c), int(r))


## 1.3 RANSAC
def count_inliers(points, circle, threshold=10) -> int:
    inliers = 0
    for point in points:
        distance = sqrt((circle[0] - point[0]) ** 2 + (circle[1] - point[1]) ** 2)
        if distance >= circle[2] - threshold and distance <= circle[2] + threshold:
            inliers += 1
    return inliers


def get_sample_points(points):
    sample_points_index = np.random.choice(
        a=[i for i in range(len(points))], size=3, replace=False
    )
    sample_points = [points[index] for index in sample_points_index]
    return sample_points


def ransac(points, img, max_iterations=10000):

    # initialize variables iteration number
    i = 0
    # initialize max count
    c = -1
    # initialize best circle
    best_xc = 0
    best_yc = 0
    best_r = 0
    best_circle = (best_xc, best_yc, best_r)
    iteration = 0
    while iteration < max_iterations:
        sample_points = get_sample_points(points)
        curr_circle = get_center(points=sample_points, img_shape=img.shape)

        # Loop to regenerate new points if they are co-linear
        while curr_circle == None:
            sample_points = get_sample_points(points)
            curr_circle = get_center(points=sample_points, img_shape=img.shape)
        inliers = count_inliers(points=points, circle=curr_circle, threshold=200)
        if inliers > c:
            printProgressBar(iteration=iteration, total=max_iterations)
            iteration += 1
            c = inliers
            best_circle = curr_circle

    return best_circle


points = get_points()
curr_circle = ransac(points=points, img=img_canny)

img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
img_canny = cv2.circle(
    img=img_canny,
    center=(curr_circle[0], curr_circle[1]),
    radius=curr_circle[2],
    color=(0, 0, 255),
    thickness=2,
)

# Rendering
window_name = "canny"
cv2.namedWindow(window_name)

while True:
    # Wait a little bit for the image to re-draw
    key = cv2.waitKey(5)
    cv2.imshow(window_name, img_canny)

    # If an x is pressed, the window will close
    if key == ord("x"):
        break
cv2.destroyAllWindows()
