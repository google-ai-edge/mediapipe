import cv2
import os
from Bounding_boxes import run
from Bounding_boxes import boxing


def ObjectDetection(video_path):

    path = os.path.abspath(video_path)

    if(video_path==0):
        cap = cv2.VideoCapture(0)
    
    else:
        cap = cv2.VideoCapture(path)

    ret = True                                       #creates a boolean 
    ret, old_frame = cap.read()                      #ret is true and the first frame of video saved in old_frame


    net = cv2.dnn.readNet('modules/object_detection/object_detection_weights.weights', 'modules/object_detection/object_detection.cfg')
        
    classes = []

    with open('modules/object_detection/object_detection_labels.txt', 'r') as f:
        classes = f.read().splitlines() 
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam/Cannot read file")

    while ret:
        ret, frame = cap.read()          #saves the first frame of video in frame

        indexes = []
        boxes = []
        class_ids = []
        confidences = []
        indexes, boxes, class_ids, confidences = run(frame, net, classes)
        font = cv2.FONT_HERSHEY_PLAIN

        if len(indexes) <= 0:    #if no bounding box
            continue
        elif len(indexes) > 0:  #if bounding box is presrnt

            frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, font)
        cv2.imshow('Output', frame)
        c = cv2.waitKey(1)           #new frame comes after () ms
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q on keyboard to stop the webcam
            break

    cap.release()
    cv2.destroyAllWindows()          #Once out of the while loop, the pop-up window closes automatically