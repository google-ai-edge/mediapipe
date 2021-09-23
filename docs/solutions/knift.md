---
layout: default
title: KNIFT (Template-based Feature Matching)
parent: Solutions
nav_order: 13
---

# MediaPipe KNIFT
{: .no_toc }

<details close markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>
---

## Overview

MediaPipe KNIFT is a template-based feature matching solution using KNIFT
(Keypoint Neural Invariant Feature Transform).

![knift_stop_sign.gif](../images/knift_stop_sign.gif)                     |
:-----------------------------------------------------------------------: |
*Fig 1. Matching a real Stop Sign with a Stop Sign template using KNIFT.* |

In many computer vision applications, a crucial building block is to establish
reliable correspondences between different views of an object or scene, forming
the foundation for approaches like template matching, image retrieval and
structure from motion. Correspondences are usually computed by extracting
distinctive view-invariant features such as
[SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) or
[ORB](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html#orb-in-opencv)
from images. The ability to reliably establish such correspondences enables
applications like image stitching to create panoramas or template matching for
object recognition in videos.

KNIFT is a general purpose local feature descriptor similar to SIFT or ORB.
Likewise, KNIFT is also a compact vector representation of local image patches
that is invariant to uniform scaling, orientation, and illumination changes.
However unlike SIFT or ORB, which were engineered with heuristics, KNIFT is an
[embedding](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)
learned directly from a large number of corresponding local patches extracted
from nearby video frames. This data driven approach implicitly encodes complex,
real-world spatial transformations and lighting changes in the embedding. As a
result, the KNIFT feature descriptor appears to be more robust, not only to
[affine distortions](https://en.wikipedia.org/wiki/Affine_transformation), but
to some degree of
[perspective distortions](https://en.wikipedia.org/wiki/Perspective_distortion_\(photography\))
as well.

For more information, please see
[MediaPipe KNIFT: Template-based feature matching](https://developers.googleblog.com/2020/04/mediapipe-knift-template-based-feature-matching.html)
in Google Developers Blog.

![template_matching_mobile_cpu.gif](../images/mobile/template_matching_android_cpu.gif) |
:-------------------------------------------------------------------------------------: |
*Fig 2. Matching US dollar bills using KNIFT.*                                          |

## Example Apps

### Matching US Dollar Bills

In MediaPipe, we've already provided an
[index file](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_index.pb)
pre-computed from the 3 template images (of US dollar bills) shown below. If
you'd like to use your own template images, see
[Matching Your Own Template Images](#matching-your-own-template-images).

![template_matching_mobile_template.jpg](../images/mobile/template_matching_mobile_template.jpg)

Please first see general instructions for
[Android](../getting_started/android.md) on how to build MediaPipe examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

*   Graph:
    [`mediapipe/graphs/template_matching/template_matching_mobile_cpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/template_matching/template_matching_mobile_cpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1tSWRfes9rAM4NrzmJBplguNQQvaeBZSa)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu:templatematchingcpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu/BUILD)

Note: MediaPipe uses OpenCV 3 by default. However, because of
[issues](https://github.com/opencv/opencv/issues/11488) between NDK 17+ and
OpenCV 3 when using
[knnMatch](https://docs.opencv.org/3.4/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89),
for this example app please use the following commands to temporarily switch to
OpenCV 4, and switch back to OpenCV 3 afterwards.

```bash
# Switch to OpenCV 4
sed -i -e 's:3.4.3/opencv-3.4.3:4.0.1/opencv-4.0.1:g' WORKSPACE
sed -i -e 's:libopencv_java3:libopencv_java4:g' third_party/opencv_android.BUILD

# Build and install app
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu
adb install -r bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu/templatematchingcpu.apk

# Switch back to OpenCV 3
sed -i -e 's:4.0.1/opencv-4.0.1:3.4.3/opencv-3.4.3:g' WORKSPACE
sed -i -e 's:libopencv_java4:libopencv_java3:g' third_party/opencv_android.BUILD
```

Tip: The example uses the TFLite
[XNNPACK delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack)
by default for faster inference. Users can change the
[option in TfLiteInferenceCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/calculators/tflite/tflite_inference_calculator.proto)
to run regular TFLite inference.

### Matching Your Own Template Images

*   Step 1: Put all template images in a single directory.

*   Step 2: To build the index file for all templates in the directory, run

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/template_matching:template_matching_tflite
    ```

    ```bash
    bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
    --calculator_graph_config_file=mediapipe/graphs/template_matching/index_building.pbtxt \
    --input_side_packets="file_directory=<template image directory>,file_suffix=png,output_index_filename=<output index filename>"
    ```

    The output index file includes the extracted KNIFT features.

*   Step 3: Replace
    [mediapipe/models/knift_index.pb](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_index.pb)
    with the index file you generated, and update
    [mediapipe/models/knift_labelmap.txt](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_labelmap.txt)
    with your own template names.

*   Step 4: Build and run the app using the same instructions in
    [Matching US Dollar Bills](#matching-us-dollar-bills).

## Resources

*   Google Developers Blog:
    [MediaPipe KNIFT: Template-based feature matching](https://developers.googleblog.com/2020/04/mediapipe-knift-template-based-feature-matching.html)
*   [Models and model cards](./models.md#knift)
