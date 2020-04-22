# Examples

Below are code samples on how to run MediaPipe on both mobile and desktop. We
currently support MediaPipe APIs on mobile for Android only but will add support
for Objective-C shortly.

## Mobile

### Hello World! on Android

[Hello World! on Android](./hello_world_android.md) should be the first mobile
Android example users go through in detail. It teaches the following:

*   Introduction of a simple MediaPipe graph running on mobile GPUs for
    [Sobel edge detection](https://en.wikipedia.org/wiki/Sobel_operator).
*   Building a simple baseline Android application that displays "Hello World!".
*   Adding camera preview support into the baseline application using the
    Android [CameraX] API.
*   Incorporating the Sobel edge detection graph to process the live camera
    preview and display the processed video in real-time.

### Hello World! on iOS

[Hello World! on iOS](./hello_world_ios.md) is the iOS version of Sobel edge
detection example.

### Object Detection with GPU

[Object Detection with GPU](./object_detection_mobile_gpu.md) illustrates how to
use MediaPipe with a TFLite model for object detection in a GPU-accelerated
pipeline.

*   [Android](./object_detection_mobile_gpu.md)
*   [iOS](./object_detection_mobile_gpu.md)

### Object Detection with CPU

[Object Detection with CPU](./object_detection_mobile_cpu.md) illustrates using
the same TFLite model in a CPU-based pipeline. This example highlights how
graphs can be easily adapted to run on CPU v.s. GPU.

### Object Detection and Tracking with GPU

[Object Detection and Tracking with GPU](./object_tracking_mobile_gpu.md) illustrates how to
use MediaPipe for object detection and tracking.

### Objectron: 3D Object Detection and Tracking with GPU

[MediaPipe Objectron is 3D Object Detection with GPU](./objectron_mobile_gpu.md)
illustrates mobile real-time 3D object detection and tracking pipeline for every
day objects like shoes and chairs

*   [Android](./objectron_mobile_gpu.md)

### Face Detection with GPU

[Face Detection with GPU](./face_detection_mobile_gpu.md) illustrates how to use
MediaPipe with a TFLite model for face detection in a GPU-accelerated pipeline.
The selfie face detection TFLite model is based on
["BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"](https://sites.google.com/view/perception-cv4arvr/blazeface),
and model details are described in the
[model card](https://sites.google.com/corp/view/perception-cv4arvr/blazeface#h.p_21ojPZDx3cqq).

*   [Android](./face_detection_mobile_gpu.md)
*   [iOS](./face_detection_mobile_gpu.md)

### Face Detection with CPU

[Face Detection with CPU](./face_detection_mobile_cpu.md) illustrates using the
same TFLite model in a CPU-based pipeline. This example highlights how graphs
can be easily adapted to run on CPU v.s. GPU.

*   [Android](./face_detection_mobile_cpu.md)
*   [iOS](./face_detection_mobile_cpu.md)

### Face Mesh with GPU

[Face Mesh with GPU](./face_mesh_mobile_gpu.md) illustrates how to run the
MediaPipe Face Mesh pipeline to perform 3D face landmark estimation in real-time
on mobile devices, utilizing GPU acceleration. The pipeline is based on
["Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs"](https://arxiv.org/abs/1907.06724),
and details of the underlying ML models are described in the
[model card](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view).

*   [Android](./face_mesh_mobile_gpu.md)
*   [iOS](./face_mesh_mobile_gpu.md)

### Hand Detection with GPU

[Hand Detection with GPU](./hand_detection_mobile_gpu.md) illustrates how to use
MediaPipe with a TFLite model for hand detection in a GPU-accelerated pipeline.

*   [Android](./hand_detection_mobile_gpu.md)
*   [iOS](./hand_detection_mobile_gpu.md)

### Hand Tracking with GPU

[Hand Tracking with GPU](./hand_tracking_mobile_gpu.md) illustrates how to use
MediaPipe with TFLite models for hand tracking in a GPU-accelerated pipeline.

*   [Android](./hand_tracking_mobile_gpu.md)
*   [iOS](./hand_tracking_mobile_gpu.md)

### Multi-Hand Tracking with GPU

[Multi-Hand Tracking with GPU](./multi_hand_tracking_mobile_gpu.md) illustrates
how to use MediaPipe with TFLite models for multi-hand tracking in a
GPU-accelerated pipeline.

*   [Android](./multi_hand_tracking_mobile_gpu.md)
*   [iOS](./multi_hand_tracking_mobile_gpu.md)

### Hair Segmentation with GPU

[Hair Segmentation on GPU](./hair_segmentation_mobile_gpu.md) illustrates how to
use MediaPipe with a TFLite model for hair segmentation in a GPU-accelerated
pipeline. The selfie hair segmentation TFLite model is based on
["Real-time Hair segmentation and recoloring on Mobile GPUs"](https://sites.google.com/view/perception-cv4arvr/hair-segmentation),
and model details are described in the
[model card](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation#h.p_NimuO7PgHxlY).

*   [Android](./hair_segmentation_mobile_gpu.md)

### Template Matching using KNIFT with CPU

[Template Matching using KNIFT on Mobile](./template_matching_mobile_cpu.md)
shows how to use MediaPipe with TFLite model for template matching using Knift
on mobile using CPU.

*   [Android](./template_matching_mobile_cpu.md)

## Desktop

### Hello World for C++

[Hello World for C++](./hello_world_desktop.md) shows how to run a simple graph
using the MediaPipe C++ APIs.

### Feature Extraction and Model Inference for YouTube-8M Challenge

[Feature Extraction and Model Inference for YouTube-8M Challenge](./youtube_8m.md)
shows how to use MediaPipe to prepare training data for the YouTube-8M Challenge
and do the model inference with the baseline model.

### Preparing Data Sets with MediaSequence

[Preparing Data Sets with MediaSequence](./media_sequence.md) shows how to use
MediaPipe for media processing to prepare video data sets for training a
TensorFlow model.

### AutoFlip - Automatic video cropping

[AutoFlip](./autoflip.md) shows how to use MediaPipe to build an automatic video
cropping pipeline that can convert an input video to arbitrary aspect ratios.

### Object Detection on Desktop

[Object Detection on Desktop](./object_detection_desktop.md) shows how to run
object detection models (TensorFlow and TFLite) using the MediaPipe C++ APIs.

[Sobel edge detection]:https://en.wikipedia.org/wiki/Sobel_operator
[CameraX]:https://developer.android.com/training/camerax

### Face Detection on Desktop with Webcam

[Face Detection on Desktop with Webcam](./face_detection_desktop.md) shows how
to use MediaPipe with a TFLite model for face detection on desktop using CPU or
GPU with live video from a webcam.

*   [Desktop GPU](./face_detection_desktop.md)
*   [Desktop CPU](./face_detection_desktop.md)

### Face Mesh on Desktop with Webcam

[Face Mesh on Desktop with Webcam](./face_mesh_desktop.md) shows how to run the
MediaPipe Face Mesh pipeline to perform 3D face landmark estimation in real-time
on desktop with webcam input.

*   [Desktop GPU](./face_mesh_desktop.md)
*   [Desktop CPU](./face_mesh_desktop.md)

### Hand Tracking on Desktop with Webcam

[Hand Tracking on Desktop with Webcam](./hand_tracking_desktop.md) shows how to
use MediaPipe with TFLite models for hand tracking on desktop using CPU or GPU
with live video from a webcam.

*   [Desktop GPU](./hand_tracking_desktop.md)
*   [Desktop CPU](./hand_tracking_desktop.md)

### Multi-Hand Tracking on Desktop with Webcam

[Multi-Hand Tracking on Desktop with Webcam](./multi_hand_tracking_desktop.md)
shows how to use MediaPipe with TFLite models for multi-hand tracking on desktop
using CPU or GPU with live video from a webcam.

*   [Desktop GPU](./multi_hand_tracking_desktop.md)
*   [Desktop CPU](./multi_hand_tracking_desktop.md)

### Hair Segmentation on Desktop with Webcam

[Hair Segmentation on Desktop with Webcam](./hair_segmentation_desktop.md) shows
how to use MediaPipe with a TFLite model for hair segmentation on desktop using
GPU with live video from a webcam.

*   [Desktop GPU](./hair_segmentation_desktop.md)

## Google Coral (ML acceleration with Google EdgeTPU)

Below are code samples on how to run MediaPipe on Google Coral Dev Board.

### Object Detection on Coral

[Object Detection on Coral with Webcam](./object_detection_coral_devboard.md)
shows how to run quantized object detection TFlite model accelerated with
EdgeTPU on
[Google Coral Dev Board](https://coral.withgoogle.com/products/dev-board).

### Face Detection on Coral

[Face Detection on Coral with Webcam](./face_detection_coral_devboard.md) shows
how to use quantized face detection TFlite model accelerated with EdgeTPU on
[Google Coral Dev Board](https://coral.withgoogle.com/products/dev-board).


## Web Browser

Below are samples that can directly be run in your web browser.
See more details in [MediaPipe on the Web](./web.md) and
[Google Developer blog post](https://mediapipe.page.link/webdevblog)

### [Face Detection In Browser](https://viz.mediapipe.dev/demo/face_detection)

### [Hand Detection In Browser](https://viz.mediapipe.dev/demo/hand_detection)

### [Hand Tracking In Browser](https://viz.mediapipe.dev/demo/hand_tracking)

### [Hair Segmentation In Browser](https://viz.mediapipe.dev/demo/hair_segmentation)
