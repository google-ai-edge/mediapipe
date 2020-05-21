# Template Matching using KNIFT on Mobile (CPU)

This doc focuses on the
[example graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/template_matching/template_matching_mobile_cpu.pbtxt)
that performs template matching with KNIFT (Keypoint Neural Invariant Feature
Transform) on mobile CPU.

![template_matching_mobile_cpu.gif](images/mobile/template_matching_android_cpu.gif)

In the visualization above, the green dots represent detected keypoints on each
frame and the red box represents the targets matched by templates using KNIFT
features (see also [model card](https://mediapipe.page.link/knift-mc)). For more
information, please see
[Google Developers Blog](https://mediapipe.page.link/knift-blog).

## Build Index Files

In MediaPipe, we've already provided a file in
[knift_index.pb](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_index.pb),
pre-computed from the 3 template images (of USD bills) shown below. If you'd
like to use your own template images, please follow the steps below, or
otherwise you can jump directly to [Android](#android).

![template_matching_mobile_template.jpg](images/mobile/template_matching_mobile_template.jpg)

### Step 1:

Put all template images in a single directory.

### Step 2:

To build the index file for all templates in the directory, run:

```bash
$ bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/template_matching:template_matching_tflite
$ bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
    --calculator_graph_config_file=mediapipe/graphs/template_matching/index_building.pbtxt \
    --input_side_packets="file_directory=<template image directory>,file_suffix=png,output_index_filename=<output index filename>"
```

The output index file includes the extracted KNIFT features.

### Step 3:

Replace
[mediapipe/models/knift_index.pb](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_index.pb)
with the index file you generated, and update
[mediapipe/models/knift_labelmap.txt](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_labelmap.txt)
with your own template names.

## Android

[Source](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu)

A prebuilt arm64 APK can be
[downloaded here](https://drive.google.com/open?id=1tSWRfes9rAM4NrzmJBplguNQQvaeBZSa).

To build and install the app yourself, run:

Note: MediaPipe uses OpenCV 3 by default. However, because of
[issues](https://github.com/opencv/opencv/issues/11488) between NDK 17+ and
OpenCV 3 when using
[knnMatch](https://docs.opencv.org/3.4/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89),
please use the following commands to temporarily switch to OpenCV 4 for the
template matching exmaple on Android, and switch back to OpenCV 3 afterwards.

```bash
# Switch to OpenCV 4
sed -i -e 's:3.4.3/opencv-3.4.3:4.0.1/opencv-4.0.1:g' WORKSPACE
sed -i -e 's:libopencv_java3:libopencv_java4:g' third_party/opencv_android.BUILD

# Build and install app
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu:templatematchingcpu
adb install -r bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu/templatematchingcpu.apk

# Switch back to OpenCV 3
sed -i -e 's:4.0.1/opencv-4.0.1:3.4.3/opencv-3.4.3:g' WORKSPACE
sed -i -e 's:libopencv_java4:libopencv_java3:g' third_party/opencv_android.BUILD
```

## Use XNNPACK Delegate

The example uses XNNPACK delegate by default. Users can change the
[option in TfLiteInferenceCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/calculators/tflite/tflite_inference_calculator.proto)
to use default TF Lite inference.

## Graph

### Main Graph

![template_matching_mobile_graph](images/mobile/template_matching_mobile_graph.png)

[Source pbtxt file](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/template_matching/template_matching_mobile_cpu.pbtxt)
