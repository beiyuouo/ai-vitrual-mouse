#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
from argparse import ArgumentParser
import math
import mediapipe as mp

parser = ArgumentParser()
parser.add_argument("-v", "--video", help="Video file to run")
parser.add_argument("-d", "--directory", help="Directory to run")
parser.add_argument("-o", "--output", default='output', help="Video file to run")


class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """
    def __init__(self, minDetectionCon=0.5):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                bboxInfo = {
                    "id": id,
                    "bbox": bbox,
                    "score": detection.score,
                    "center": (cx, cy)
                }
                bboxs.append(bboxInfo)
                if draw:
                    img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 0, 255), 2)
        return img, bboxs


def from_video(args):
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        if bboxs:
            # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def from_image(args):
    for filename in os.listdir(args.directory):
        img = cv2.imread(os.path.join(args.directory, filename))
        detector = FaceDetector()
        img, bboxs = detector.findFaces(img)

        if bboxs:
            # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

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
