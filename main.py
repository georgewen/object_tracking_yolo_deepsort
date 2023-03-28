# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import random
import cv2
import numpy as np
import json
from ultralytics import YOLO

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.image_encoder import *
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

max_cosine_distance = 0.4
nn_budget = None
encoder_model_filename = 'model_data/mars-small128.pb'
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

tracker = Tracker(metric)
results = []

# start main

video_path = os.path.join('.', 'data', 'test3.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()


# Find OpenCV version

fps = cap.get(cv2.CAP_PROP_FPS)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
 

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
frame_idx = 1

encoder = create_box_encoder(encoder_model_filename, batch_size=1)

while ret:

    objects = model(frame)

    for obj in objects:
        detections = []
        for r in obj.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        
        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]
        
        features = encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))


        # Update tracker.
        tracker.predict()
        tracker.update(dets)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                if len([res[0] for res in results if res[1]==track.track_id]) >= 1:
                    duration = (frame_idx - min([res[0] for res in results if res[1]==track.track_id]))/fps
                    print("car " + str(track.track_id) + " exits in " + f'{duration:.2f}' + " seconds") 
                    if duration > 1.5:
                        bbox = track.to_tlbr()
                        track_id = track.track_id
                        x1,y1,x2,y2 = bbox
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)                                                     
                continue
            bbox = track.to_tlbr()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        

        for track in tracker.tracks:
            bbox = track.to_tlbr()
            track_id = track.track_id
            x1,y1,x2,y2 = bbox
            #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            frame = cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow('frame',frame)
    cv2.waitKey(25)

    cap_out.write(frame)
    ret, frame = cap.read()
    frame_idx =  frame_idx + 1

# json_string = json.dumps(results)
# with open("Output.txt", "w") as text_file:
#     text_file.write(json_string)

cap.release()
cap_out.release()
cv2.destroyAllWindows()
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

# convert list to pandas dataframe
df = pd.DataFrame(results, columns=["frameIdx","track_id","x1","y1","x2","y2"])
df2 = df.groupby('track_id').agg(
    frame_min_idx = pd.NamedAgg(column="frameIdx", aggfunc="min"),
    frame_max_idx = pd.NamedAgg(column="frameIdx",aggfunc="max"))
df2["duration"] = df2["frame_max_idx"]/30 - df2["frame_min_idx"]/30

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df2["duration"])