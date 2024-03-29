# Copyright 2022 The MediaPipe Authors.
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

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_tasks_vision_aar")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

android_library(
    name = "core",
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    exports = [
        ":core_java",
        ":libmediapipe_tasks_vision_jni_lib",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
    ],
)

android_library(
    name = "core_java",
    srcs = glob(["core/*.java"]),
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        "//mediapipe/framework/formats:rect_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:core_java",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe tasks vision per task AAR.
filegroup(
    name = "core_java_src",
    srcs = glob(["core/**/*.java"]),
    visibility = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:__subpackages__"],
)

# The native library of all MediaPipe vision tasks.
cc_binary(
    name = "libmediapipe_tasks_vision_jni.so",
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location //mediapipe/tasks/java:version_script.lds)",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/vision/face_detector:face_detector_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
        "//mediapipe/tasks/cc/vision/face_stylizer:face_stylizer_graph",
        "//mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer_graph",
        "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_graph",
        "//mediapipe/tasks/cc/vision/image_classifier:image_classifier_graph",
        "//mediapipe/tasks/cc/vision/image_embedder:image_embedder_graph",
        "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_graph",
        "//mediapipe/tasks/cc/vision/interactive_segmenter:interactive_segmenter_graph",
        "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
        "//mediapipe/tasks/java:version_script.lds",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
)

cc_library(
    name = "libmediapipe_tasks_vision_jni_lib",
    srcs = [":libmediapipe_tasks_vision_jni.so"],
    alwayslink = 1,
)

android_library(
    name = "objectdetector",
    srcs = [
        "objectdetector/ObjectDetectionResult.java",
        "objectdetector/ObjectDetector.java",
        "objectdetector/ObjectDetectorResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "objectdetector/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:detection_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:detection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "imageclassifier",
    srcs = [
        "imageclassifier/ImageClassifier.java",
        "imageclassifier/ImageClassifierResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "imageclassifier/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_classifier/proto:image_classifier_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classificationresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "facestylizer",
    srcs = [
        "facestylizer/FaceStylizer.java",
        "facestylizer/FaceStylizerResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "imagesegmenter/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_stylizer/proto:face_stylizer_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "gesturerecognizer",
    srcs = [
        "gesturerecognizer/GestureRecognizer.java",
        "gesturerecognizer/GestureRecognizerResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "gesturerecognizer/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_classifier_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_recognizer_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:hand_gesture_recognizer_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/processors:classifieroptions",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "poselandmarker",
    srcs = [
        "poselandmarker/PoseLandmarker.java",
        "poselandmarker/PoseLandmarkerResult.java",
        "poselandmarker/PoseLandmarksConnections.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "poselandmarker/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/pose_detector/proto:pose_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:connection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "handlandmarker",
    srcs = [
        "handlandmarker/HandLandmark.java",
        "handlandmarker/HandLandmarker.java",
        "handlandmarker/HandLandmarkerResult.java",
        "handlandmarker/HandLandmarksConnections.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "handlandmarker/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:connection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "imagesegmenter",
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "imagesegmenter/AndroidManifest.xml",
    exports = [
        ":core",
        ":imagesegmenter_java",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
    ],
    deps = ["@maven//:com_google_guava_guava"],
)

android_library(
    name = "imagesegmenter_java",
    srcs = [
        "imagesegmenter/ImageSegmenter.java",
        "imagesegmenter/ImageSegmenterResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "imagesegmenter/AndroidManifest.xml",
    deps = [
        ":core_java",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:core_java",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe tasks vision per task AAR.
filegroup(
    name = "image_segmenter_java_src",
    srcs = glob(["imagesegmenter/**/*.java"]),
    visibility = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:__subpackages__"],
)

android_library(
    name = "interactivesegmenter",
    srcs = [
        "imagesegmenter/ImageSegmenterResult.java",
        "interactivesegmenter/InteractiveSegmenter.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "interactivesegmenter/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalizedkeypoint",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/util:color_java_proto_lite",
        "//mediapipe/util:render_data_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "imageembedder",
    srcs = [
        "imageembedder/ImageEmbedder.java",
        "imageembedder/ImageEmbedderResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "imageembedder/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_embedder/proto:image_embedder_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embedding",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embeddingresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:cosinesimilarity",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "facedetector",
    srcs = [
        "facedetector/FaceDetector.java",
        "facedetector/FaceDetectorResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "facedetector/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:detection_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:detection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "facelandmarker",
    srcs = [
        "facelandmarker/FaceLandmarker.java",
        "facelandmarker/FaceLandmarkerResult.java",
        "facelandmarker/FaceLandmarksConnections.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "facedetector/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/framework/formats:matrix_data_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:connection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe tasks vision per task AAR.
filegroup(
    name = "face_landmarker_java_src",
    srcs = glob(["facelandmarker/**/*.java"]),
    visibility = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:__subpackages__"],
)

android_library(
    name = "holisticlandmarker",
    srcs = [
        "holisticlandmarker/HolisticLandmarker.java",
        "holisticlandmarker/HolisticLandmarkerResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "facedetector/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/pose_detector/proto:pose_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:any_java_proto",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_tasks_vision_aar(
    name = "tasks_vision",
    srcs = glob(
        ["**/*.java"],
        exclude = ["imagegenerator/**"],
    ),
    native_library = ":libmediapipe_tasks_vision_jni_lib",
)
