#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   HandDetector.py 
@Time    :   2021-11-23 15:59:45 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(detectionCon=0.8)

while True:
    ret, frame = cap.read()
    hand, image = detector.findHands(frame)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break