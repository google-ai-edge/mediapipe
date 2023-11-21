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

import mediapipe as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_path', required=False, default="/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4", type=str, help='Path to a video file')
parser.add_argument('--output_video_path', required=False, default="/mediapipe/object_output.mp4", type=str, help='Output path to a video file')
args = parser.parse_args()

source = args.input_video_path
output = args.output_video_path

ovms_object_detection = mp.solutions.ovms_object_detection
with ovms_object_detection.OvmsObjectDetection(side_inputs=
        {'input_video_path':source,
         'output_video_path':output}) as ovms_object_detection:
        results = ovms_object_detection.process()
