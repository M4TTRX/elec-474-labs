from __future__ import print_function
import cv2
import argparse


class FaceFinder:
    def __init__(self, path) -> None:

        # load frontal face detector files
        cv2.samples.addSamplesDataSearchPath(path)
        front_face_cascade_name = "haarcascade_frontalface_default.xml"

        # load frontal alt detector files
        cv2.samples.addSamplesDataSearchPath(path)
        alt_face_cascade_name = "haarcascade_frontalface_alt.xml"

        # load frontal alt2 detector files
        cv2.samples.addSamplesDataSearchPath(path)
        alt2_face_cascade_name = "haarcascade_frontalface_alt2.xml"

        # load frontal profile detector files
        cv2.samples.addSamplesDataSearchPath(path)
        profile_face_cascade_name = "haarcascade_profileface.xml"

        # create and load the frontal face cascade
        self.frontal_face_cascade = cv2.CascadeClassifier()
        if not self.frontal_face_cascade.load(
            cv2.samples.findFile(front_face_cascade_name)
        ):
            print("--(!)Error loading frontal face cascade")
            exit(0)

        # create and load the alt face cascade
        self.alt_face_cascade = cv2.CascadeClassifier()
        if not self.alt_face_cascade.load(cv2.samples.findFile(alt_face_cascade_name)):
            print("--(!)Error loading alt face cascade")
            exit(0)

        # create and load the alt2 face cascade
        self.alt2_face_cascade = cv2.CascadeClassifier()
        if not self.alt2_face_cascade.load(
            cv2.samples.findFile(alt2_face_cascade_name)
        ):
            print("--(!)Error loading alt2 face cascade")
            exit(0)

        # create and load the profile face cascade
        self.profile_face_cascade = cv2.CascadeClassifier()
        if not self.profile_face_cascade.load(
            cv2.samples.findFile(profile_face_cascade_name)
        ):
            print("--(!)Error loading profile face cascade")
            exit(0)

    def ultimate_find_face(self, frame):
        face = self.find_frontal_face(frame)
        if face is not None:
            return face
        face = self.find_alt_face(frame)
        if face is not None:
            return face
        face = self.find_alt2_face(frame)
        if face is not None:
            return face
        face = self.find_profile_face(frame)
        return face

    def find_frontal_face(self, frame):
        # get the first face in the image
        faces = self.frontal_face_cascade.detectMultiScale(frame)
        if len(faces) == 0:
            return None
        face = self.frontal_face_cascade.detectMultiScale(frame)[0]
        x, y, width, height = face

        # crop the picture so it only includes that face
        face = frame[y : y + height, x : x + width]
        return face

    def find_alt_face(self, frame):
        # get the first face in the image
        faces = self.alt_face_cascade.detectMultiScale(frame)
        if len(faces) == 0:
            return None
        face = self.alt_face_cascade.detectMultiScale(frame)[0]
        x, y, width, height = face

        # crop the picture so it only includes that face
        face = frame[y : y + height, x : x + width]
        return face

    def find_alt2_face(self, frame):
        # get the first face in the image
        faces = self.alt2_face_cascade.detectMultiScale(frame)
        if len(faces) == 0:
            return None
        face = self.alt2_face_cascade.detectMultiScale(frame)[0]
        x, y, width, height = face

        # crop the picture so it only includes that face
        face = frame[y : y + height, x : x + width]
        return face

    def find_profile_face(self, frame):
        # get the first face in the image
        faces = self.profile_face_cascade.detectMultiScale(frame)
        if len(faces) == 0:
            return None
        face = self.profile_face_cascade.detectMultiScale(frame)[0]
        x, y, width, height = face

        # crop the picture so it only includes that face
        face = frame[y : y + height, x : x + width]
        return face

    def detect_triplet(self, triplet):
        center, left, right = triplet
        center_face = self.ultimate_find_face(center)
        left_face = self.ultimate_find_face(left)
        right_face = self.ultimate_find_face(right)
        return [
            center_face,
            left_face,
            right_face,
        ]
