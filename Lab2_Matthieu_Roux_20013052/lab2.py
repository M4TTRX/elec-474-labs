import numpy as np
import cv2
import copy
import math
import time

# terminal colours
TGREEN = "\033[32m"  # Green character
TYELLOW = "\033[93m"  # Yellow text character
ENDC = "\033[m"  # reset to the defaults

img_path = "Lab2_Matthieu_Roux_20013052/b.jpg"
img = cv2.imread(img_path)

img_height = img.shape[0]
img_width = img.shape[1]
supbixel_range = img.shape[2]


# generate a k, the number of clusters (colors) we will have
max_k_value = 15  # we will set a max value for k so that we don't generate an out of control amount of clusters
k = np.random.randint(low=2, high=max_k_value)


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


def euclidean_distance(p1, p2):
    """
    euclidean_distance(element_1, element_2)
        Returns the euclidean distance between point 1 and 2 point 1 and 2 are 2 arrays of length n. n represents the dimension of points 1 and 2.
    """
    p1_int = [int(attribute) for attribute in p1]
    p2_int = [int(attribute) for attribute in p2]
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1_int, p2_int)]))


def choose_region(pixel, cluster_centers):
    euclidean_distances = [
        euclidean_distance(cluster_center, pixel) for cluster_center in cluster_centers
    ]
    minval = min(euclidean_distances)
    index = euclidean_distances.index(minval)
    return index


def get_voronoi_average(cluster_centers):
    region_sizes = np.zeros((k, supbixel_range))
    region_totals = np.zeros((k, supbixel_range))
    region_map = np.zeros((img_height, img_width), np.uint8)

    iteration = 0
    total_steps = img_height * img_width
    for index in np.ndindex(img.shape[:2]):
        # The region index is the index the of the cluster region in cluster_centers
        region_index = choose_region(img[index], cluster_centers)

        # Add this region to the region map
        region_map[index] = region_index

        # region_sizes keeps tabs on how many pixels there are in each region
        region_sizes[region_index] = region_sizes[region_index] + 1

        # region totals will be later divided by the region_sizes a region's mean value
        region_totals[region_index] = np.add(region_totals[region_index], img[index])

        # progress bar stuff, it's necessary, trust me
        if iteration % 1000 == 0 or iteration == total_steps:
            printProgressBar(iteration=iteration, total=total_steps)
        iteration = iteration + 1
        means = [
            np.uint8(total / size) for total, size in zip(region_totals, region_sizes)
        ]
    return means, region_map


def compute_clusters(k, threshold=30, max_loops=3):
    # compute cluster centers
    cluster_indexes = [
        (
            np.random.randint(low=0, high=img_height),
            np.random.randint(low=0, high=img_width),
        )
        for i in range(k)
    ]
    cluster_centers = [img[index] for index in cluster_indexes]

    for loop in range(max_loops):
        # get the Voronoid regions
        region_means, region_map = get_voronoi_average(cluster_centers)

        if is_within_threshold(threshold, cluster_centers, region_means):
            print(
                "\n\nSegmentation took ", TGREEN + str(loop + 1), ENDC, " runs", end=""
            )
            return cluster_centers, region_map
        print(
            ENDC + "\nMeans do not match the threshold yet. Currently in loop: ",
            str(loop + 1),
            end="",
        )
        # if we are not within the threshold update the cluster_center values with means
        cluster_centers = region_means
    return cluster_centers, region_map


def is_within_threshold(threshold, cluster_centers, region_means):
    for center, region in zip(cluster_centers, region_means):
        if euclidean_distance(center, region) > threshold:
            return False
    return True


def update_img(k, img):
    # Printing messages
    print(
        "Computing regions... Please wait\n\nWe will divide the image in ",
        TGREEN + str(k),
        ENDC,
        " colors!",
        end="",
    )
    start = time.time()
    print()

    cluster_centers, pixel_to_region_map = compute_clusters(k)
    for index in np.ndindex(pixel_to_region_map.shape[:2]):
        img[index] = cluster_centers[pixel_to_region_map[index]]

    # Print end of computation fanfare
    end = time.time()
    print("Computation time:\t", int(end - start), "s\n")
    return img


# reset all is used to reset the image
def reset_all():
    global img
    img = cv2.imread(img_path)


# update image

img = update_img(k, img)

# Rendering
window_name = img_path
cv2.namedWindow(window_name)

# on_trackbar is called when the trackbar is changed, it updates the threshold
def on_trackbar(val):
    global k, img
    k = val
    reset_all()
    img = update_img(k, img)


# create trackbar
cv2.createTrackbar("k", window_name, k, max_k_value, on_trackbar)
while True:
    # Wait a little bit for the image to re-draw
    key = cv2.waitKey(5)
    cv2.imshow(window_name, img)

    # If an x is pressed, the window will close
    if key == ord("x"):
        break