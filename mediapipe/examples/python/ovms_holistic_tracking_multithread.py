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

class RequestingThread(threading.Thread):
    def __init__(self, index):
        print(f"Initializing requesting thread index: {index}")
        super().__init__()
        self.index = index
        self.input_frame = None
        self.output_frame = None
        self.predict_durations = []
        self.input_ready_event = threading.Event()
        self.output_ready_event = threading.Event()

    def is_initialized(self):
        return not (self.input_frame is None and self.output_frame is None)

    def wait_for_input(self):
        self.input_ready_event.wait()
        self.input_ready_event.clear()

    def wait_for_result(self):
        self.output_ready_event.wait()
        self.output_ready_event.clear()

    def notify_input_ready(self):
        self.input_ready_event.set()

    def notify_output_ready(self):
        self.output_ready_event.set()

    def set_input(self, frame):
        self.input_frame = frame
        self.notify_input_ready()

    def get_output(self):
        return self.output_frame

    def get_average_latency(self):
        return np.average(np.array(self.predict_durations))

    def run(self):
        print(f"Launching requesting thread index: {self.index}")
        global force_exit
        ovms_holistic_tracking = mp.solutions.ovms_holistic_tracking
        with ovms_holistic_tracking.OvmsHolisticTracking() as ovms_holistic_tracking:
            while (True):
                self.wait_for_input()
                if force_exit:
                        print("Detected exit signal...")
                        break

                predict_start_time = time.time()
                result = ovms_holistic_tracking.process(self.input_frame)
                
                predict_duration = time.time() - predict_start_time
                predict_duration *= 1000
                self.predict_durations.append(predict_duration)

                if result is None:
                    self.output_frame = np.array(self.input_frame, copy=True)
                else:
                    # output_video is the output_stream name from the graph
                    self.output_frame = result.output_video

                self.notify_output_ready()
                print(f"Stopping requesting thread index: {self.index}")

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', required=False, default=4, type=int, help='Number of threads for parallel service requesting')
parser.add_argument('--input_video_path', required=False, default="/mediapipe/video.mp4", type=str, help='Camera ID number or path to a video file')
parser.add_argument('--output_video_path', required=False, default="holistic_output.mp4", type=str, help='Output path to a video file')
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
force_exit = False
    
threads = [RequestingThread(i) for i in range(args.num_threads)]

for thread in threads:
    thread.start()

def finish():
    global force_exit
    force_exit = True
    for thread in threads:
        thread.notify_input_ready()
        thread.join()

def grab_frame(cap):
    success, frame = cap.read()
    if not success:
        print("[WARNING] No Input frame")
        finish()
        return None
    return frame

thread_id = 0
frames_processed = 0
last_display_time = time.time()
app_start_time = time.time()

if grab_frame(cap) is None:
    print("[ERROR] Check camera input...")
    force_exit = True

while not force_exit:
    if not threads[thread_id].is_initialized():
        threads[thread_id].set_input(grab_frame(cap))
        thread_id = (thread_id + 1) % args.num_threads
        continue

    threads[thread_id].wait_for_result()
    avg_latency_for_thread = threads[thread_id].get_average_latency()
    frame_to_display = threads[thread_id].get_output()
    threads[thread_id].set_input(grab_frame(cap))

    out.write(frame_to_display)
    now = time.time()
    time_since_last_display = now - last_display_time
    last_display_time = now

    frames_processed += 1

    current_fps = 1 / (time_since_last_display if time_since_last_display > 0 else 1)
    avg_fps = 1 / ((now - app_start_time) / frames_processed)
    
    print(f"ThreadID: {thread_id:3}; Current FPS: {current_fps:8.2f}; Average FPS: {avg_fps:8.2f}; Average latency: {avg_latency_for_thread:8.2f}ms")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        finish()
        break

    thread_id = (thread_id + 1) % args.num_threads

total_time = time.time() - app_start_time
print(f"Total processing time: {total_time:8.2f}s")
finish()
# When everything done, release the capture
cap.release()
out.release()
print(f"Finished.")
