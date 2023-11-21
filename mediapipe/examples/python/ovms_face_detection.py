# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import cv2
import mediapipe as mp
import time
import numpy as np
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_path', required=False, default="/mediapipe/video.mp4", type=str, help='Camera ID number or path to a video file')
parser.add_argument('--output_video_path', required=False, default="face_output.mp4", type=str, help='Output path to a video file')
args = parser.parse_args()

try:
    source = int(args.input_video_path)
except ValueError:
    source = args.input_video_path

output = args.output_video_path

cap = cv2.VideoCapture(source)
out = cv2.VideoWriter()
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Input file FPS - %s width - %s height - %s" % (fps, width, height ))
out.open(output, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
if not out.isOpened():
    print("Output file open error: " + str(out.isOpened()))
    exit()

print("Output file opened for writing: " + str(out.isOpened()))

def grab_frame(cap):
    success, frame = cap.read()
    if not success:
        print("[WARNING] No Input frame")
        return None
    return frame

input_frame = grab_frame(cap)
if  input_frame is None:
    print("[ERROR] Check camera or file input...")
    exit(-1)

ovms_face_detection = mp.solutions.ovms_face_detection
with ovms_face_detection.OvmsFaceDetection() as ovms_face_detection:
    while input_frame is not None:      
        result = ovms_face_detection.process(input_frame)
        if result is None:
            output_frame = np.array(input_frame, copy=True)
        else:
            # output_video is the output_stream name from the graph
            output_frame = result.output_video

        out.write(output_frame)

        input_frame = grab_frame(cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
out.release()
print(f"Finished.")
