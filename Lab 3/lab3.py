from math import dist, sqrt
import numpy as np
import cv2
from numpy.core.fromnumeric import size

# terminal colours
TGREEN = "\033[32m"  # Green character
TYELLOW = "\033[93m"  # Yellow text character
TRED = "\033[91m"  # Red text character
ENDC = "\033[m"  # reset to the defaults

img_url = "Lab 3/parliament_clock.jpg"
img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2GRAY)

threshold_low = 250
threshold_hi = 255

# use edge detection on img
img_canny = cv2.Canny(img, threshold1=threshold_low, threshold2=threshold_hi)


def sobel_edge_detection(img):
    sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = cv2.filter2D(img, 5, sobel_x_filter)
    sobel_y = cv2.filter2D(img, 5, sobel_y_filter)
    return sobel_x, sobel_y


img = cv2.GaussianBlur(src=img, ksize=(0, 0), sigmaX=3)
i_x, i_y = sobel_edge_detection(img)


## 1.4 Get the Gradient Magnitude G(i,j) of the image by combining i_x and i_y


def gradient_magnitude(i_x, i_y):
    gradient_magnitude = np.zeros(img.shape)
    for index in np.ndindex(img.shape):
        gradient_magnitude[index] = abs(i_x[index]) + abs(i_y[index])
    gradient_magnitude *= 255 / np.amax(gradient_magnitude)
    return gradient_magnitude.astype(np.uint8)


gm = gradient_magnitude(i_x, i_y)

## 1.5 Thresholding
threshold = 18


def apply_threshold(img, threshold):
    output_img = (img > threshold) * 255
    return output_img.astype(np.uint8)


def get_points(image):
    return np.argwhere(image == 255)


## 1.2 Point Circle Fitting


def get_center(points, img_shape, verbose=False, safety=True, max_size=None):
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

    if vector1[1] == 0 or vector2[1] == 0 or vector2[0] == 0 or vector1[0] == 0:
        return None

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

    if y_c > img_shape[0] or x_c > img_shape[1] and safety:
        if verbose:
            print("Center of the circle is outside the bounds of the image")
        return None
    ## Calculate the radius
    r = sqrt((x_c - point1[0]) ** 2 + (y_c - point1[1]) ** 2)
    # if max_size and r > max_size:
    #     return None
    return (int(x_c), int(y_c), int(r))


def get_distance(point1, point2):
    distance = sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return distance


## 1.3 RANSAC
def count_inliers(points, circle, threshold=10):
    inlier_count = 0
    inliers = []
    for point in points:
        distance = get_distance([circle[1], circle[0]], point)
        if distance >= circle[2] - threshold and distance <= circle[2] + threshold:
            inlier_count += 1
            inliers.append(point)
    return inlier_count, inliers


def get_sample_points(points):
    sample_points_index = np.random.randint(0, len(points), size=3)
    sample_points = [points[index] for index in sample_points_index]
    return sample_points


def ransac(points, img, max_no_improvement_rounds=10, max_iterations=100):

    # initialize variables iteration number
    i = 0
    # initialize max count
    c = -1
    # initialize best circle
    best_circle = (0, 0, 0)
    best_inliers = []
    rounds_since_last_improvement = 0
    iterations = 0
    while rounds_since_last_improvement <= max_no_improvement_rounds:

        iterations += 1
        if iterations > max_iterations:
            break

        # sample 3 points to test a circle
        sample_points = get_sample_points(points)

        # compute the circle for those 3 points
        curr_circle = get_center(points=sample_points, img_shape=img.shape)

        # loop to regenerate new points if they are co-linear or other errors are found
        while curr_circle == None:
            sample_points = get_sample_points(points)
            curr_circle = get_center(points=sample_points, img_shape=img.shape)
        inliers_count, inliers = count_inliers(
            points=points, circle=curr_circle, threshold=2
        )

        # Update the best circle if best inlier ratio is found
        if inliers_count / (curr_circle[2] ** 0.5) > c:
            c = inliers_count / (curr_circle[2] ** 0.5)
            best_circle = curr_circle
            best_inliers = inliers
            rounds_since_last_improvement = 0
        else:
            rounds_since_last_improvement += 1
    return best_circle, best_inliers, iterations


# 1.4 Post Processing
def post_process_circle(curr_circle, inliers):
    # The centroid of the inliers
    inlier_y = int(sum([inlier[0] for inlier in inliers]) / len(inliers))
    inlier_x = int(sum([inlier[1] for inlier in inliers]) / len(inliers))

    radius = int(
        sum([get_distance((inlier_y, inlier_x), inlier) for inlier in inliers])
        / len(inliers)
    )
    return (inlier_x, inlier_y, radius)


def create_circle(image, color, circle):
    output_img = cv2.circle(
        img=image,
        center=(int(circle[0]), int(circle[1])),
        radius=int(circle[2]),
        color=color,
        thickness=2,
    )
    return output_img


sobel_img = apply_threshold(gm, 120)
points = get_points(sobel_img)
og_circle, inliers, iterations = ransac(
    points=points, img=sobel_img, max_no_improvement_rounds=200, max_iterations=2000
)

# 1.4 Post Processing
refined_circle = post_process_circle(og_circle, inliers)

# Display circle
img = cv2.imread(img_url)
img = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR)
img = create_circle(img, (0, 0, 255), og_circle)
img = create_circle(img, (0, 255, 0), refined_circle)

for inlier in inliers:
    img[inlier[0], inlier[1]] = (0, 255, 255)

print(
    f"{TYELLOW}==={ENDC} Runtime statistics {TYELLOW}==={ENDC}\n",
    f"{TRED}Original Circle{ENDC}:\ty = {og_circle[0]}\ty = {og_circle[1]}\tradius = {og_circle[2]}\n",
    f"{TGREEN}Refined Circle{ENDC}: \ty = {refined_circle[0]}\ty = {refined_circle[1]}\tradius = {refined_circle[2]}\n\n",
    f"Number of iterations: {iterations}\n",
    end="",
)
# Rendering
window_name = "circles"
cv2.namedWindow(window_name)

while True:
    # Wait a little bit for the image to re-draw
    key = cv2.waitKey(5)
    cv2.imshow(window_name, img)

    # If an x is pressed, the window will close
    if key == ord("x"):
        break
cv2.destroyAllWindows()
