---
layout: default
title: Face Mesh
parent: Solutions
nav_order: 2
---

# MediaPipe Face Mesh
{: .no_toc }

1. TOC
{:toc}
---

## Overview

MediaPipe Face Mesh is a face geometry solution that estimates 468 3D face
landmarks in real-time even on mobile devices. It employs machine learning (ML)
to infer the 3D surface geometry, requiring only a single camera input without
the need for a dedicated depth sensor. Utilizing lightweight model architectures
together with GPU acceleration throughout the pipeline, the solution delivers
real-time performance critical for live experiences. The core of the solution is
the same as what powers
[YouTube Stories](https://youtube-creators.googleblog.com/2018/11/introducing-more-ways-to-share-your.html)'
creator effects, the
[Augmented Faces API in ARCore](https://developers.google.com/ar/develop/java/augmented-faces/)
and the
[ML Kit Face Contour Detection API](https://firebase.google.com/docs/ml-kit/face-detection-concepts#contours).

![face_mesh_ar_effects.gif](../images/face_mesh_ar_effects.gif) |
:-------------------------------------------------------------: |
*Fig 1. AR effects utilizing facial surface geometry.*          |

## ML Pipeline

Our ML pipeline consists of two real-time deep neural network models that work
together: A detector that operates on the full image and computes face locations
and a 3D face landmark model that operates on those locations and predicts the
approximate surface geometry via regression. Having the face accurately cropped
drastically reduces the need for common data augmentations like affine
transformations consisting of rotations, translation and scale changes. Instead
it allows the network to dedicate most of its capacity towards coordinate
prediction accuracy. In addition, in our pipeline the crops can also be
generated based on the face landmarks identified in the previous frame, and only
when the landmark model could no longer identify face presence is the face
detector invoked to relocalize the face. This strategy is similar to that
employed in our [MediaPipe Hands](./hands.md) solution, which uses a palm detector
together with a hand landmark model.

The pipeline is implemented as a MediaPipe
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_mesh/face_mesh_mobile.pbtxt)
that uses a
[face landmark subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark_front_gpu.pbtxt)
from the
[face landmark module](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark),
and renders using a dedicated
[face renderer subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_mesh/subgraphs/face_renderer_gpu.pbtxt).
The
[face landmark subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark_front_gpu.pbtxt)
internally uses a
[face_detection_subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection/face_detection_front_gpu.pbtxt)
from the
[face detection module](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection).

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

## Models

### Face Detection Model

The face detector is the same [BlazeFace](https://arxiv.org/abs/1907.05047)
model used in [MediaPipe Face Detection](./face_detection.md). Please refer to
[MediaPipe Face Detection](./face_detection.md) for details.

### Face Landmark Model

For 3D face landmarks we employed transfer learning and trained a network with
several objectives: the network simultaneously predicts 3D landmark coordinates
on synthetic rendered data and 2D semantic contours on annotated real-world
data. The resulting network provided us with reasonable 3D landmark predictions
not just on synthetic but also on real-world data.

The 3D landmark network receives as input a cropped video frame without
additional depth input. The model outputs the positions of the 3D points, as
well as the probability of a face being present and reasonably aligned in the
input. A common alternative approach is to predict a 2D heatmap for each
landmark, but it is not amenable to depth prediction and has high computational
costs for so many points. We further improve the accuracy and robustness of our
model by iteratively bootstrapping and refining predictions. That way we can
grow our dataset to increasingly challenging cases, such as grimaces, oblique
angle and occlusions.

You can find more information about the face landmark model in this
[paper](https://arxiv.org/abs/1907.06724).

![face_mesh_android_gpu.gif](../images/mobile/face_mesh_android_gpu.gif)   |
:------------------------------------------------------------------------: |
*Fig 2. Output of MediaPipe Face Mesh: the red box indicates the cropped area as input to the landmark model, the red dots represent the 468 landmarks in 3D, and the green lines connecting landmarks illustrate the contours around the eyes, eyebrows, lips and the entire face.* |

## Example Apps

Please first see general instructions for
[Android](../getting_started/building_examples.md#android), [iOS](../getting_started/building_examples.md#ios) and
[desktop](../getting_started/building_examples.md#desktop) on how to build MediaPipe examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Mobile

*   Graph:
    [`mediapipe/graphs/face_mesh/face_mesh_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_mesh/face_mesh_mobile.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1pUmd7CXCL_onYMbsZo5p91cH0oNnR4gi)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/facemeshgpu:facemeshgpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facemeshgpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/facemeshgpu:FaceMeshGpuApp`](http:/mediapipe/examples/ios/facemeshgpu/BUILD)

Tip: Maximum number of faces to detect/process is set to 1 by default. To change
it, for Android modify `NUM_FACES` in
[MainActivity.java](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facemeshgpu/MainActivity.java),
and for iOS modify `kNumFaces` in
[ViewController.mm](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/facemeshgpu/ViewController.mm).

### Desktop

*   Running on CPU
    *   Graph:
        [`mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/face_mesh:face_mesh_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/face_mesh/BUILD)
*   Running on GPU
    *   Graph:
        [`mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/face_mesh:face_mesh_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/face_mesh/BUILD)

Tip: Maximum number of faces to detect/process is set to 1 by default. To change
it, in the graph file modify the option of `ConstantSidePacketCalculator`.

## Resources

*   Google AI Blog:
    [Real-Time AR Self-Expression with Machine Learning](https://ai.googleblog.com/2019/03/real-time-ar-self-expression-with.html)
*   TensorFlow Blog:
    [Face and hand tracking in the browser with MediaPipe and TensorFlow.js](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)
*   Paper:
    [Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs](https://arxiv.org/abs/1907.06724)
    ([poster](https://docs.google.com/presentation/d/1-LWwOMO9TzEVdrZ1CS1ndJzciRHfYDJfbSxH_ke_JRg/present?slide=id.g5986dd4b4c_4_212))
*   Face detection model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection/face_detection_front.tflite)
*   Face landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/facemesh/1)
*   [Model card](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view)
