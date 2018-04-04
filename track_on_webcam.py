# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf


from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import colorsys

from tools.generate_detections import create_box_encoder

from application_util.visualization import create_unique_color_float,create_unique_color_uchar

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths

video_capture = cv2.VideoCapture('test1.mp4')
#video_capture = cv2.VideoCapture(0)


metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", 0.2, None)
tracker = Tracker(metric)


def dectect(frame):

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, w, h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return pick


encoder = create_box_encoder('resources/networks/mars-small128.pb')

while True:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy

    ret, image = video_capture.read()
    image = imutils.resize(image, width=min(400, image.shape[1]))
    pick = dectect(image)
    detections = []

    features = encoder(image, pick)

    for i,(x, y, w, h) in enumerate(pick):
    	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detections.append(Detection([x,y,w,h],1, features[i,:]))

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        color = create_unique_color_uchar(track.track_id)
        coor = track.to_tlwh().astype(np.int)
        cv2.rectangle(image, (coor[0], coor[1]), (coor[0] + coor[2], coor[1] + coor[3]), color, 2,
                      )


    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
