from __future__ import print_function
import cv2
import argparse


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(
            frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4
        )
        faceROI = frame_gray[y : y + h, x : x + w]
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv2.imshow("Capture - Face detection", frame)


img = cv2.imread("Final_Project/lib/test.jpg")

parser = argparse.ArgumentParser(
    description="Code for Cascade Classifier tutorial. Inspired by the openCV tutorial"
)
parser.add_argument(
    "--face_cascade",
    help="Path to face cascade.",
    default="data/haarcascades/haarcascade_frontalface_alt.xml",
)
parser.add_argument(
    "--eyes_cascade",
    help="Path to eyes cascade.",
    default="data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
# -- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("--(!)Error loading face cascade")
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print("--(!)Error loading eyes cascade")
    exit(0)


detectAndDisplay(img)