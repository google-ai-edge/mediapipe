---
layout: default
title: Hands
parent: Solutions
nav_order: 4
---

# MediaPipe Hands
{: .no_toc }

1. TOC
{:toc}
---

## Overview

The ability to perceive the shape and motion of hands can be a vital component
in improving the user experience across a variety of technological domains and
platforms. For example, it can form the basis for sign language understanding
and hand gesture control, and can also enable the overlay of digital content and
information on top of the physical world in augmented reality. While coming
naturally to people, robust real-time hand perception is a decidedly challenging
computer vision task, as hands often occlude themselves or each other (e.g.
finger/palm occlusions and hand shakes) and lack high contrast patterns.

MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs
machine learning (ML) to infer 21 3D landmarks of a hand from just a single
frame. Whereas current state-of-the-art approaches rely primarily on powerful
desktop environments for inference, our method achieves real-time performance on
a mobile phone, and even scales to multiple hands. We hope that providing this
hand perception functionality to the wider research and development community
will result in an emergence of creative use cases, stimulating new applications
and new research avenues.

![hand_tracking_3d_android_gpu.gif](../images/mobile/hand_tracking_3d_android_gpu.gif) |
:------------------------------------------------------------------------------------: |
*Fig 1. Tracked 3D hand landmarks are represented by dots in different shades, with the brighter ones denoting landmarks closer to the camera.* |

## ML Pipeline

MediaPipe Hands utilizes an ML pipeline consisting of multiple models working
together: A palm detection model that operates on the full image and returns an
oriented hand bounding box. A hand landmark model that operates on the cropped
image region defined by the palm detector and returns high-fidelity 3D hand
keypoints. This strategy is similar to that employed in our
[MediaPipe Face Mesh](./face_mesh.md) solution, which uses a face detector
together with a face landmark model.

Providing the accurately cropped hand image to the hand landmark model
drastically reduces the need for data augmentation (e.g. rotations, translation
and scale) and instead allows the network to dedicate most of its capacity
towards coordinate prediction accuracy. In addition, in our pipeline the crops
can also be generated based on the hand landmarks identified in the previous
frame, and only when the landmark model could no longer identify hand presence
is palm detection invoked to relocalize the hand.

The pipeline is implemented as a MediaPipe
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt),
which internally utilizes a
[palm/hand detection subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/subgraphs/hand_detection_gpu.pbtxt),
a
[hand landmark subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/subgraphs/hand_landmark_gpu.pbtxt)
and a
[renderer subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/subgraphs/renderer_gpu.pbtxt).

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

## Models

### Palm Detection Model

To detect initial hand locations, we designed a
[single-shot detector](https://arxiv.org/abs/1512.02325) model optimized for
mobile real-time uses in a manner similar to the face detection model in
[MediaPipe Face Mesh](./face_mesh.md). Detecting hands is a decidedly complex
task: our
[model](https://github.com/google/mediapipe/tree/master/mediapipe/models/palm_detection.tflite) has
to work across a variety of hand sizes with a large scale span (~20x) relative
to the image frame and be able to detect occluded and self-occluded hands.
Whereas faces have high contrast patterns, e.g., in the eye and mouth region,
the lack of such features in hands makes it comparatively difficult to detect
them reliably from their visual features alone. Instead, providing additional
context, like arm, body, or person features, aids accurate hand localization.

Our method addresses the above challenges using different strategies. First, we
train a palm detector instead of a hand detector, since estimating bounding
boxes of rigid objects like palms and fists is significantly simpler than
detecting hands with articulated fingers. In addition, as palms are smaller
objects, the non-maximum suppression algorithm works well even for two-hand
self-occlusion cases, like handshakes. Moreover, palms can be modelled using
square bounding boxes (anchors in ML terminology) ignoring other aspect ratios,
and therefore reducing the number of anchors by a factor of 3-5. Second, an
encoder-decoder feature extractor is used for bigger scene context awareness
even for small objects (similar to the RetinaNet approach). Lastly, we minimize
the focal loss during training to support a large amount of anchors resulting
from the high scale variance.

With the above techniques, we achieve an average precision of 95.7% in palm
detection. Using a regular cross entropy loss and no decoder gives a baseline of
just 86.22%.

### Hand Landmark Model

After the palm detection over the whole image our subsequent hand landmark
[model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark.tflite)
performs precise keypoint localization of 21 3D hand-knuckle coordinates inside
the detected hand regions via regression, that is direct coordinate prediction.
The model learns a consistent internal hand pose representation and is robust
even to partially visible hands and self-occlusions.

To obtain ground truth data, we have manually annotated ~30K real-world images
with 21 3D coordinates, as shown below (we take Z-value from image depth map, if
it exists per corresponding coordinate). To better cover the possible hand poses
and provide additional supervision on the nature of hand geometry, we also
render a high-quality synthetic hand model over various backgrounds and map it
to the corresponding 3D coordinates.

| ![hand_crops.png](../images/mobile/hand_crops.png)                          |
| :-------------------------------------------------------------------------: |
| *Fig 2. Top: Aligned hand crops passed to the tracking network with ground truth annotation. Bottom: Rendered synthetic hand images with ground truth annotation.* |

## Example Apps

Please first see general instructions for
[Android](../getting_started/building_examples.md#android), [iOS](../getting_started/building_examples.md#ios)
and [desktop](../getting_started/building_examples.md#desktop) on how to build MediaPipe
examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Mobile

#### Main Example

*   Graph:
    [`mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1uCjS0y0O0dTDItsMh8x2cf4-l3uHW1vE)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu:handtrackinggpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/handtrackinggpu/BUILD)

#### With Multi-hand Support

*   Graph:
    [`mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1Wk6V9EVaz1ks_MInPqqVGvvJD01SGXDc)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/multihandtrackinggpu:multihandtrackinggpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/multihandtrackinggpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/multihandtrackinggpu:MultiHandTrackingGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/multihandtrackinggpu/BUILD)

There are two key differences between this graph and that in the
[main example](#main-example) (which handles only one hand):

1.  There is a `NormalizedRectVectorHasMinSize` calculator, that checks if in
    input vector of `NormalizedRect` objects has a minimum size equal to `N`. In
    this graph, if the vector contains fewer than `N` objects,
    `MultiHandDetection` subgraph runs. Otherwise, the `GateCalculator` doesn't
    send any image packets to the `MultiHandDetection` subgraph. This way, the
    main graph is efficient in that it avoids running the costly hand detection
    step when there are already `N` hands in the frame.
2.  The `MergeCalculator` has been replaced by the `AssociationNormRect`
    calculator. This `AssociationNormRect` takes as input a vector of
    `NormalizedRect` objects from the `MultiHandDetection` subgraph on the
    current frame, and a vector of `NormalizedRect` objects from the
    `MultiHandLandmark` subgraph from the previous frame, and performs an
    association operation between these objects. This calculator ensures that
    the output vector doesn't contain overlapping regions based on the specified
    `min_similarity_threshold`.

#### Palm/Hand Detection Only (no landmarks)

*   Graph:
    [`mediapipe/graphs/hand_tracking/hand_detection_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_detection_mobile.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1qUlTtH7Ydg-wl_H6VVL8vueu2UCTu37E)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/handdetectiongpu:handdetectiongpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handdetectiongpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/handdetectiongpu:HandDetectionGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/handdetectiongpu/BUILD)

### Desktop

#### Main Example

*   Running on CPU
    *   Graph:
        [`mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/hand_tracking/BUILD)
*   Running on GPU
    *   Graph:
        [`mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/hand_tracking/BUILD)

#### With Multi-hand Support

*   Running on CPU
    *   Graph:
        [`mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live)
    *   Target:
        [`mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/multi_hand_tracking/BUILD)
*   Running on GPU
    *   Graph:
        [`mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/multi_hand_tracking/BUILD)

### Web

Please refer to [these instructions](../index.md#mediapipe-on-the-web).

## Resources

*   Google AI Blog:
    [On-Device, Real-Time Hand Tracking with MediaPipe](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)
*   TensorFlow Blog:
    [Face and hand tracking in the browser with MediaPipe and TensorFlow.js](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)
*   Paper:
    [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214)
    ([presentation](https://www.youtube.com/watch?v=I-UOrvxxXEk))
*   Palm detection model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/palm_detection.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/handdetector/1)
*   Hand landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/handskeleton/1)
*   [Model card](https://mediapipe.page.link/handmc)
