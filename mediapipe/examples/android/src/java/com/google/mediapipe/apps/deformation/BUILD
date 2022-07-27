# Copyright 2019 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/graphs/deformation:mobile_calculators",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

ASSETS_DIR = "//mediapipe/graphs/deformation/config"

android_binary(
    name = "deformationgpu",
    srcs = glob(["*.java"]),
    assets = [
        "//mediapipe/graphs/deformation:mobile_gpu.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        ASSETS_DIR+":triangles.txt",
        ASSETS_DIR+":index_names.txt",
        ASSETS_DIR+":mediumFaceIndexes.txt",
        ASSETS_DIR+":mediumNoseIndexes.txt",
        ASSETS_DIR+":noseAllIndexes.txt",
        ASSETS_DIR+":cheekbonesIndexes.txt",
        ASSETS_DIR+":additionalNoseIndexes1.txt",
        ASSETS_DIR+":additionalNoseIndexes2.txt",
        ASSETS_DIR+":additionalNoseIndexes3.txt",
        ASSETS_DIR+":faceOvalIndexes.txt",
        ASSETS_DIR+":lowerLipCnt.txt",
        ASSETS_DIR+":upperLipCnt.txt",
        ASSETS_DIR+":widerLowerLipPts1.txt",
        ASSETS_DIR+":widerUpperLipPts1.txt",        
    ],
    assets_dir = "",
    manifest = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.deformationgpu",
        "appName": "Deformation",
        "mainActivity": "com.google.mediapipe.apps.basic.MainActivity",
        "cameraFacingFront": "True",
        "binaryGraphName": "mobile_gpu.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "2",
    },
    multidex = "native",
    deps = [
        ":mediapipe_jni_lib",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:basic_lib",
         "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/modules/face_geometry/protos:face_geometry_java_proto_lite",
    ],
)

