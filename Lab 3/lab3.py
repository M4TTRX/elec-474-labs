from math import sqrt
import numpy as np
import cv2


img = cv2.cvtColor(cv2.imread("Lab 3/circle.jpg"), cv2.COLOR_BGR2GRAY)

threshold_low = 50
threshold_hi = 150

# use edge detection on img
img_canny = cv2.Canny(img, threshold1=threshold_low, threshold2=threshold_hi)


def get_points():
    return np.argwhere(img_canny == 255)


points = get_points()

## 3 Point Circle Fitting
def get_center(point1, point2, point3):
    vector1 = (point2[0] - point1[0], point2[1] - point1[1])
    vector2 = (point3[0] - point1[0], point3[1] - point1[1])

    # Use the vectors to calculate the mid points between the point1 -> point 2
    # and point1 -> point3
    mid_1 = (point1[0] + vector1[0] / 2, point1[1] + vector1[1] / 2)
    mid_2 = (point1[0] + vector2[0] / 2, point1[1] + vector2[1] / 2)

    # Calculate line slopes m_1 and m_2
    m_1 = vector1[1] / vector1[0]
    m_2 = vector2[1] / vector2[0]

    # Find perpendicular slopes m_per_1 and m_per_2
    m_per_1 = -1 / m_1 if m_1 != 0 else float("inf")
    m_per_2 = -1 / m_2 if m_2 != 0 else float("inf")

    # Calculate the x of center of the circle
    x_c = (mid_2[1] - mid_2[0] * m_per_2 - (mid_1[1] - mid_1[0] * m_per_1)) / (
        m_per_1 - m_per_2
    )

    # Now that we have x we can find y
    y_c = x_c * m_per_1 + (mid_1[1] - mid_1[0] * m_per_1)

    ## Calculate the radius
    r = sqrt((x_c - point1[0]) ** 2 + (y_c - point1[1]) ** 2)
    return ((x_c, y_c), r)


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
