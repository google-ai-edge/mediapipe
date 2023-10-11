---
layout: forward
target: https://developers.google.com/mediapipe/solutions/guide#legacy
title: Box Tracking
parent: MediaPipe Legacy Solutions
nav_order: 10
---

# MediaPipe Box Tracking
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

**Attention:** *Thank you for your interest in MediaPipe Solutions.
We have ended support for this MediaPipe Legacy Solution as of March 1, 2023.
For more information, see the
[MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/guide#legacy)
site.*

----

## Overview

MediaPipe Box Tracking has been powering real-time tracking in
[Motion Stills](https://ai.googleblog.com/2016/12/get-moving-with-new-motion-stills.html),
[YouTube's privacy blur](https://youtube-creators.googleblog.com/2016/02/blur-moving-objects-in-your-video-with.html),
and [Google Lens](https://lens.google.com/) for several years, leveraging
classic computer vision approaches.

The box tracking solution consumes image frames from a video or camera stream,
and starting box positions with timestamps, indicating 2D regions of interest to
track, and computes the tracked box positions for each frame. In this specific
use case, the starting box positions come from object detection, but the
starting position can also be provided manually by the user or another system.
Our solution consists of three main components: a motion analysis component, a
flow packager component, and a box tracking component. Each component is
encapsulated as a MediaPipe calculator, and the box tracking solution as a whole
is represented as a MediaPipe
[subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/subgraphs/box_tracking_gpu.pbtxt).

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/).

In the
[box tracking subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/subgraphs/box_tracking_gpu.pbtxt),
the MotionAnalysis calculator extracts features (e.g. high-gradient corners)
across the image, tracks those features over time, classifies them into
foreground and background features, and estimates both local motion vectors and
the global motion model. The FlowPackager calculator packs the estimated motion
metadata into an efficient format. The BoxTracker calculator takes this motion
metadata from the FlowPackager calculator and the position of starting boxes,
and tracks the boxes over time. Using solely the motion data (without the need
for the RGB frames) produced by the MotionAnalysis calculator, the BoxTracker
calculator tracks individual objects or regions while discriminating from
others. Please see
[Object Detection and Tracking using MediaPipe](https://developers.googleblog.com/2019/12/object-detection-and-tracking-using-mediapipe.html)
in Google Developers Blog for more details.

An advantage of our architecture is that by separating motion analysis into a
dedicated MediaPipe calculator and tracking features over the whole image, we
enable great flexibility and constant computation independent of the number of
regions tracked! By not having to rely on the RGB frames during tracking, our
tracking solution provides the flexibility to cache the metadata across a batch
of frame. Caching enables tracking of regions both backwards and forwards in
time; or even sync directly to a specified timestamp for tracking with random
access.

## Object Detection and Tracking

MediaPipe Box Tracking can be paired with ML inference, resulting in valuable
and efficient pipelines. For instance, box tracking can be paired with ML-based
object detection to create an object detection and tracking pipeline. With
tracking, this pipeline offers several advantages over running detection per
frame (e.g., [MediaPipe Object Detection](./object_detection.md)):

*   It provides instance based tracking, i.e. the object ID is maintained across
    frames.
*   Detection does not have to run every frame. This enables running heavier
    detection models that are more accurate while keeping the pipeline
    lightweight and real-time on mobile devices.
*   Object localization is temporally consistent with the help of tracking,
    meaning less jitter is observable across frames.

![object_tracking_android_gpu.gif](https://mediapipe.dev/images/mobile/object_tracking_android_gpu.gif) |
:----------------------------------------------------------------------------------: |
*Fig 1. Box tracking paired with ML-based object detection.*                         |

The object detection and tracking pipeline can be implemented as a MediaPipe
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/object_detection_tracking_mobile_gpu.pbtxt),
which internally utilizes an
[object detection subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/subgraphs/object_detection_gpu.pbtxt),
an
[object tracking subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/subgraphs/object_tracking_gpu.pbtxt),
and a
[renderer subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/subgraphs/renderer_gpu.pbtxt).

In general, the object detection subgraph (which performs ML model inference
internally) runs only upon request, e.g. at an arbitrary frame rate or triggered
by specific signals. More specifically, in this particular
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/object_detection_tracking_mobile_gpu.pbtxt)
a PacketResampler calculator temporally subsamples the incoming video frames to
0.5 fps before they are passed into the object detection subgraph. This frame
rate can be configured differently as an option in PacketResampler.

The object tracking subgraph runs in real-time on every incoming frame to track
the detected objects. It expands the
[box tracking subgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/subgraphs/box_tracking_gpu.pbtxt)
with additional functionality: when new detections arrive it uses IoU
(Intersection over Union) to associate the current tracked objects/boxes with
new detections to remove obsolete or duplicated boxes.

## Example Apps

Please first see general instructions for
[Android](../getting_started/android.md), [iOS](../getting_started/ios.md) and
[desktop](../getting_started/cpp.md) on how to build MediaPipe examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Mobile

Note: Object detection is using TensorFlow Lite on GPU while tracking is on CPU.

*   Graph:
    [`mediapipe/graphs/tracking/object_detection_tracking_mobile_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/object_detection_tracking_mobile_gpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1UXL9jX4Wpp34TsiVogugV3J3T9_C5UK-)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/objecttrackinggpu:objecttrackinggpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objecttrackinggpu/BUILD)
*   iOS target: Not available

### Desktop

*   Running on CPU (both for object detection using TensorFlow Lite and
    tracking):
    *   Graph:
        [`mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/object_tracking:object_tracking_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/object_tracking/BUILD)
*   Running on GPU: Not available

## Resources

*   Google Developers Blog:
    [Object Detection and Tracking using MediaPipe](https://developers.googleblog.com/2019/12/object-detection-and-tracking-using-mediapipe.html)
*   Google AI Blog:
    [Get moving with the new Motion Stills](https://ai.googleblog.com/2016/12/get-moving-with-new-motion-stills.html)
*   YouTube Creator Blog: [Blur moving objects in your video with the new Custom
    blurring tool on
    YouTube](https://youtube-creators.googleblog.com/2016/02/blur-moving-objects-in-your-video-with.html)
