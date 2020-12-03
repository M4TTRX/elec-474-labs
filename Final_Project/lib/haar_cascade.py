from __future__ import print_function
import cv2
import argparse


class FaceFinder:
    def __init__(self, path) -> None:

        # set proper path to look for xml files
        cv2.samples.addSamplesDataSearchPath(path)
        face_cascade_name = "data/haarcascades/haarcascade_frontalface_alt.xml"

        # create and load the face cascade
        self.face_cascade = cv2.CascadeClassifier()
        if not self.face_cascade.load(cv2.samples.findFile(face_cascade_name)):
            print("--(!)Error loading face cascade")
            exit(0)

    def find_face(self, frame):
        # get the first face in the image
        faces = self.face_cascade.detectMultiScale(frame)
        if len(faces) == 0:
            return None
        face = self.face_cascade.detectMultiScale(frame)[0]
        x, y, width, height = face

        # crop the picture so it only includes that face
        face = frame[y : y + height, x : x + width]
        return face


img = cv2.imread("Final_Project/lib/test.jpg")
