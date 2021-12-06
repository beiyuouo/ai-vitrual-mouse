#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from cvzone.HandTrackingModule import HandDetector
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", "--video", help="Video file to run")
parser.add_argument("-d", "--directory", help="Directory to run")
parser.add_argument("-o", "--output", default='output', help="Video file to run")


def from_video(args):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1['center']
            handType1 = hand1["type"]

            fingers1 = detector.fingersUp(hand1)

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                centerPoint2 = hand2['center']
                handType2 = hand2["type"]

                fingers2 = detector.fingersUp(hand2)

                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                length, info = detector.findDistance(lmList1[8], lmList2[8])

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def from_image(args):
    for filename in os.listdir(args.directory):
        img = cv2.imread(os.path.join(args.directory, filename))
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        hands, img = detector.findHands(img)
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1['center']
            handType1 = hand1["type"]

            fingers1 = detector.fingersUp(hand1)

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                centerPoint2 = hand2['center']
                handType2 = hand2["type"]

                fingers2 = detector.fingersUp(hand2)

                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                length, info = detector.findDistance(lmList1[8], lmList2[8])

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
