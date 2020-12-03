import cv2


def show_img(imgs, names=[], use_plt=False):

    # create window names if there are no names / it is invalid
    if len(names) == 0 or len(names) != len(imgs):
        names = [str(i) for i in range(len(imgs))]
    # create windows
    for name in names:
        cv2.namedWindow(name)

    while True:
        # wait a little bit for the image to re-draw
        key = cv2.waitKey(5)
        for img, name in zip(imgs, names):
            if img != None:
                cv2.imshow(name, img)

        # if an x is pressed, the window will close
        if key == ord("x"):
            break
    cv2.destroyAllWindows()