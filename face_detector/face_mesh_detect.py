#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
from argparse import ArgumentParser
import math
import mediapipe as mp

parser = ArgumentParser()
parser.add_argument("-v", "--video", action='store_true', help="Video file to run")
parser.add_argument("-d", "--directory", help="Directory to run")
parser.add_argument("-o", "--output", default='output', help="Output directory")


class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def from_video(args):
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        # if faces:
        #    print(faces[0])

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def from_image(args):
    for filename in os.listdir(args.directory):
        img = cv2.imread(os.path.join(args.directory, filename))
        detector = FaceMeshDetector(maxFaces=2)
        img, faces = detector.findFaceMesh(img)
        #if faces:
        #    print(faces[0])

        cv2.imshow("Image", img)
        cv2.imwrite(os.path.join(args.output, filename), img)
        cv2.waitKey(1)


def main():
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.video:
        from_video(args)
    elif args.directory:
        from_image(args)
    else:
        from_video(args)


if __name__ == '__main__':
    main()
