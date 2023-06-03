import numpy as np
import random
from ultralytics import YOLO
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tools.generate_detections import create_box_encoder
from deep_sort.deep_sort.detection import Detection
import cv2
import os



def annotate(model, file_parth, filename):

    # initiating DeepSort's tracker
    similarity_metric = NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker = Tracker(similarity_metric)

    # Feature extractor
    feature_extractor = create_box_encoder("./deep_sort/mars-small128.pb", batch_size=1)
    color_tracks = {}

    # loading the appropriate openCV parser and video generator
    cap = cv2.VideoCapture(file_parth)
    ret, frame = cap.read()

    ret_file_path = "outcoming/" + filename
    ret_file = open(ret_file_path, "x")
    ret_file.close()
    cap_out = cv2.VideoWriter(ret_file_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                              (frame.shape[1], frame.shape[0]))

    # Getting the YOLO results of the first frame
    resultss = model(frame)

    while ret:
        for results in resultss:

            # Generating a list of Detection(s) for the current frame
            bboxes, features, scores = ([] for k in range(3))
            for bbox_wrapper in results.boxes.data.tolist():
                min_x, min_y, max_x, max_y, score, class_id = bbox_wrapper
                tlwh = (min_x, min_y, max_x - min_x, max_y - min_y)
                bboxes.append(np.asarray(tlwh))
                scores.append(score)
            features = feature_extractor(frame, bboxes)

            detections = []
            for k in range(len(bboxes)):
                detections.append(Detection(bboxes[k], scores[k], features[k]))

            # Feeding the Detection(s) to DeepSort's Tracker
            tracker.predict()
            tracker.update(detections)

            # Adding the assigned (colored) detections to the frame
            for track in tracker.tracks:
                if track.track_id not in color_tracks:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    while color in color_tracks.values():
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    color_tracks[track.track_id] = color

                cv2.rectangle(frame, (int(track.to_tlbr()[0]), int(track.to_tlbr()[1])),
                              (int(track.to_tlbr()[2]), int(track.to_tlbr()[3])), color_tracks[track.track_id], 3)
                # print(track.to_tlbr())

        # Adding the updated frame to the output video
        cap_out.write(frame)

        # Getting the next frame
        ret, frame = cap.read()
        resultss = model(frame)

    cap.release()
    cap_out.release()

    return ret_file_path


