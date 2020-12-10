---
layout: default
title: Objectron (3D Object Detection)
parent: Solutions
nav_order: 11
---

# MediaPipe Objectron
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

MediaPipe Objectron is a mobile real-time 3D object detection solution for
everyday objects. It detects objects in 2D images, and estimates their poses
through a machine learning (ML) model, trained on the [Objectron dataset](https://github.com/google-research-datasets/Objectron).

![objectron_shoe_android_gpu.gif](../images/mobile/objectron_shoe_android_gpu.gif) | ![objectron_chair_android_gpu.gif](../images/mobile/objectron_chair_android_gpu.gif) | ![objectron_camera_android_gpu.gif](../images/mobile/objectron_camera_android_gpu.gif) | ![objectron_cup_android_gpu.gif](../images/mobile/objectron_cup_android_gpu.gif)
:--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------:
*Fig 1a. Shoe Objectron*                                                           | *Fig 1b. Chair Objectron*                                                            | *Fig 1c. Camera Objectron*                                                             | *Fig 1d. Cup Objectron*

Object detection is an extensively studied computer vision problem, but most of
the research has focused on
[2D object prediction](https://ai.googleblog.com/2017/06/supercharge-your-computer-vision-models.html).
While 2D prediction only provides 2D bounding boxes, by extending prediction to
3D, one can capture an object’s size, position and orientation in the world,
leading to a variety of applications in robotics, self-driving vehicles, image
retrieval, and augmented reality. Although 2D object detection is relatively
mature and has been widely used in the industry, 3D object detection from 2D
imagery is a challenging problem, due to the lack of data and diversity of
appearances and shapes of objects within a category.

![objectron_example_results.png](../images/objectron_example_results.png) |
:-----------------------------------------------------------------------: |
*Fig 2. Objectron example results.*                                       |

## Obtaining Real-World 3D Training Data

While there are ample amounts of 3D data for street scenes, due to the
popularity of research into self-driving cars that rely on 3D capture sensors
like LIDAR, datasets with ground truth 3D annotations for more granular everyday
objects are extremely limited. To overcome this problem, we developed a novel
data pipeline using mobile augmented reality (AR) session data. With the arrival
of [ARCore](https://developers.google.com/ar) and
[ARKit](https://developer.apple.com/augmented-reality/),
[hundreds of millions](https://arinsider.co/2019/05/13/arcore-reaches-400-million-devices/)
of smartphones now have AR capabilities and the ability to capture additional
information during an AR session, including the camera pose, sparse 3D point
clouds, estimated lighting, and planar surfaces.

In order to label ground truth data, we built a novel annotation tool for use
with AR session data, which allows annotators to quickly label 3D bounding boxes
for objects. This tool uses a split-screen view to display 2D video frames on
which are overlaid 3D bounding boxes on the left, alongside a view showing 3D
point clouds, camera positions and detected planes on the right. Annotators draw
3D bounding boxes in the 3D view, and verify its location by reviewing the
projections in 2D video frames. For static objects, we only need to annotate an
object in a single frame and propagate its location to all frames using the
ground truth camera pose information from the AR session data, which makes the
procedure highly efficient.

| ![objectron_data_annotation.gif](../images/objectron_data_annotation.gif)    |
| :--------------------------------------------------------------------------: |
| *Fig 3. Real-world data annotation for 3D object detection. (Right) 3D bounding boxes are annotated in the 3D world with detected surfaces and point clouds. (Left) Projections of annotated 3D bounding boxes are overlaid on top of video frames making it easy to validate the annotation.* |

## AR Synthetic Data Generation

A popular approach is to complement real-world data with synthetic data in order
to increase the accuracy of prediction. However, attempts to do so often yield
poor, unrealistic data or, in the case of photorealistic rendering, require
significant effort and compute. Our novel approach, called AR Synthetic Data
Generation, places virtual objects into scenes that have AR session data, which
allows us to leverage camera poses, detected planar surfaces, and estimated
lighting to generate placements that are physically probable and with lighting
that matches the scene. This approach results in high-quality synthetic data
with rendered objects that respect the scene geometry and fit seamlessly into
real backgrounds. By combining real-world data and AR synthetic data, we are
able to increase the accuracy by about 10%.

![objectron_synthetic_data_generation.gif](../images/objectron_synthetic_data_generation.gif) |
:-------------------------------------------------------------------------------------------: |
*Fig 4. An example of AR synthetic data generation. The virtual white-brown cereal box is rendered into the real scene, next to the real blue book.* |

## ML Pipelines for 3D Object Detection

We built two ML pipelines to predict the 3D bounding box of an object from a
single RGB image: one is a two-stage pipeline and the other is a single-stage
pipeline. The two-stage pipeline is 3x faster than the single-stage pipeline
with similar or better accuracy. The single stage pipeline is good at detecting
multiple objects, whereas the two stage pipeline is good for a single dominant
object.

### Two-stage Pipeline

Our two-stage pipeline is illustrated by the diagram in Fig 5. The first stage
uses an object detector to find the 2D crop of the object. The second stage
takes the image crop and estimates the 3D bounding box. At the same time, it
also computes the 2D crop of the object for the next frame, such that the object
detector does not need to run every frame.

![objectron_network_architecture.png](../images/objectron_2stage_network_architecture.png) |
:----------------------------------------------------------------------------------------: |
*Fig 5. Network architecture and post-processing for two-stage 3D object detection.*       |

We can use any 2D object detector for the first stage. In this solution, we use
[TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) trained
with the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html).
The second stage 3D bounding box predictor we released runs 83FPS on Adreno 650
mobile GPU.

### Single-stage Pipeline

![objectron_network_architecture.png](../images/objectron_network_architecture.png) |
:---------------------------------------------------------------------------------: |
*Fig 6. Network architecture and post-processing for single-stage 3D object detection.* |

Our [single-stage pipeline](https://arxiv.org/abs/2003.03522) is illustrated by
the diagram in Fig 6, the model backbone has an encoder-decoder architecture,
built upon
[MobileNetv2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html).
We employ a multi-task learning approach, jointly predicting an object's shape
with detection and regression. The shape task predicts the object's shape
signals depending on what ground truth annotation is available, e.g.
segmentation. This is optional if there is no shape annotation in training data.
For the detection task, we use the annotated bounding boxes and fit a Gaussian
to the box, with center at the box centroid, and standard deviations
proportional to the box size. The goal for detection is then to predict this
distribution with its peak representing the object’s center location. The
regression task estimates the 2D projections of the eight bounding box vertices.
To obtain the final 3D coordinates for the bounding box, we leverage a well
established pose estimation algorithm
([EPnP](https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/)). It
can recover the 3D bounding box of an object, without a priori knowledge of the
object dimensions. Given the 3D bounding box, we can easily compute pose and
size of the object. The model is light enough to run real-time on mobile devices
(at 26 FPS on an Adreno 650 mobile GPU).

![objectron_sample_network_results.png](../images/objectron_sample_network_results.png) |
:-------------------------------------------------------------------------------------: |
*Fig 7. Sample results of our network — (Left) original 2D image with estimated bounding boxes, (Middle) object detection by Gaussian distribution, (Right) predicted segmentation mask.* |

#### Detection and Tracking

When the model is applied to every frame captured by the mobile device, it can
suffer from jitter due to the ambiguity of the 3D bounding box estimated in each
frame. To mitigate this, we adopt the same detection+tracking strategy in our
[2D object detection and tracking pipeline](./box_tracking.md#object-detection-and-tracking)
in [MediaPipe Box Tracking](./box_tracking.md). This mitigates the need to run
the network on every frame, allowing the use of heavier and therefore more
accurate models, while keeping the pipeline real-time on mobile devices. It also
retains object identity across frames and ensures that the prediction is
temporally consistent, reducing the jitter.

The Objectron 3D object detection and tracking pipeline is implemented as a
MediaPipe
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/object_occlusion_tracking_1stage.pbtxt),
which internally uses a
[detection subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/objectron_detection_1stage_gpu.pbtxt)
and a
[tracking subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/objectron_tracking_1stage_gpu.pbtxt).
The detection subgraph performs ML inference only once every few frames to
reduce computation load, and decodes the output tensor to a FrameAnnotation that
contains nine keypoints: the 3D bounding box's center and its eight vertices.
The tracking subgraph runs every frame, using the box traker in
[MediaPipe Box Tracking](./box_tracking.md) to track the 2D box tightly
enclosing the projection of the 3D bounding box, and lifts the tracked 2D
keypoints to 3D with
[EPnP](https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/). When
new detection becomes available from the detection subgraph, the tracking
subgraph is also responsible for consolidation between the detection and
tracking results, based on the area of overlap.

## Objectron Dataset

We also released our [Objectron dataset](http://objectron.dev), with which we
trained our 3D object detection models. The technical details of the Objectron
dataset, including usage and tutorials, are available on
the [dataset website](https://github.com/google-research-datasets/Objectron/).

## Example Apps

Please first see general instructions for
[Android](../getting_started/android.md) and [iOS](../getting_started/ios.md) on
how to build MediaPipe examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Two-stage Objectron

*   Graph:
    [`mediapipe/graphs/object_detection_3d/object_occlusion_tracking.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/object_occlusion_tracking.pbtxt)

*   Android target:
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/BUILD).

    Build for **shoes** (default) with:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/file/d/1ANW9WDOCb8QO1r8gDC03A4UgrPkICdPP/view?usp=sharing)

    ```bash
    bazel build -c opt --config android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
    ```

    Build for **chairs** with:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/file/d/1lcUv1TBnv_SxnKSQwdOqbdLa9mkaTJHy/view?usp=sharing)

    ```bash
    bazel build -c opt --config android_arm64 --define chair=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
    ```

    Build for **cups** with:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/file/d/1bf77KDkowwrduleiC9B1M1XnEhjnOQbX/view?usp=sharing)

    ```bash
    bazel build -c opt --config android_arm64 --define cup=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
    ```

    Build for **cameras** with:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/file/d/1GM7lPO-s5URVxIzQur1bLsionEJs3yIl/view?usp=sharing)

    ```bash
    bazel build -c opt --config android_arm64 --define camera=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
    ```

*   iOS target: Not available

### Single-stage Objectron

*   Graph:
    [`mediapipe/graphs/object_detection_3d/object_occlusion_tracking_1stage.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/object_occlusion_tracking.pbtxt)

*   Android target:
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/BUILD).

    Build with **single-stage** model for **shoes** with:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/file/d/1MvaEg4dkvKN8jAU1Z2GtudyXi1rQHYsE/view?usp=sharing)

    ```bash
    bazel build -c opt --config android_arm64 --define shoe_1stage=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
    ```

    Build with **single-stage** model for **chairs** with:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/file/d/1GJL4z3jr-wD1jMHGd4NBfOG-Yoq5t167/view?usp=sharing)

    ```bash
    bazel build -c opt --config android_arm64 --define chair_1stage=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
    ```

*   iOS target: Not available

## Resources

*   Google AI Blog:
    [Announcing the Objectron Dataset](https://ai.googleblog.com/2020/11/announcing-objectron-dataset.html)
*   Google AI Blog:
    [Real-Time 3D Object Detection on Mobile Devices with MediaPipe](https://ai.googleblog.com/2020/03/real-time-3d-object-detection-on-mobile.html)
*   Paper: [MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak
    Shape Supervision](https://arxiv.org/abs/2003.03522)
*   Paper:
    [Instant 3D Object Tracking with Applications in Augmented Reality](https://drive.google.com/open?id=1O_zHmlgXIzAdKljp20U_JUkEHOGG52R8)
    ([presentation](https://www.youtube.com/watch?v=9ndF1AIo7h0))
*   [Models and model cards](./models.md#objectron)
