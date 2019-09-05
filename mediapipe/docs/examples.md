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

### Face Detection with GPU

[Face Detection with GPU](./face_detection_mobile_gpu.md) illustrates how to use
MediaPipe with a TFLite model for face detection in a GPU-accelerated pipeline.
The selfie face detection TFLite model is based on
["BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"](https://sites.google.com/view/perception-cv4arvr/blazeface),
and model details are described in the
[model card](https://sites.google.com/corp/view/perception-cv4arvr/blazeface#h.p_21ojPZDx3cqq).

*   [Android](./face_detection_mobile_gpu.md)
*   [iOS](./face_detection_mobile_gpu.md)

### Hand Detection with GPU

[Hand Detection with GPU](./hand_detection_mobile_gpu.md) illustrates how to use
MediaPipe with a TFLite model for hand detection in a GPU-accelerated pipeline.

*   [Android](./hand_detection_mobile_gpu.md)
*   [iOS](./hand_detection_mobile_gpu.md)

### Hand Tracking with GPU

[Hand Tracking with GPU](./hand_tracking_mobile_gpu.md) illustrates how to use
MediaPipe with a TFLite model for hand tracking in a GPU-accelerated pipeline.

*   [Android](./hand_tracking_mobile_gpu.md)
*   [iOS](./hand_tracking_mobile_gpu.md)

### Hair Segmentation with GPU

[Hair Segmentation on GPU](./hair_segmentation_mobile_gpu.md) illustrates how to
use MediaPipe with a TFLite model for hair segmentation in a GPU-accelerated
pipeline. The selfie hair segmentation TFLite model is based on
["Real-time Hair segmentation and recoloring on Mobile GPUs"](https://sites.google.com/view/perception-cv4arvr/hair-segmentation),
and model details are described in the
[model card](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation#h.p_NimuO7PgHxlY).

*   [Android](./hair_segmentation_mobile_gpu.md)

## Desktop

### Hello World for C++

[Hello World for C++](./hello_world_desktop.md) shows how to run a simple graph
using the MediaPipe C++ APIs.

### Feature Extration for YouTube-8M Challenge

[Feature Extration for YouTube-8M Challenge](./youtube_8m.md) shows how to use
MediaPipe to prepare training data for the YouTube-8M Challenge.

### Preparing Data Sets with MediaSequence

[Preparing Data Sets with MediaSequence](./media_sequence.md) shows how to use
MediaPipe for media processing to prepare video data sets for training a
TensorFlow model.

### Object Detection on Desktop

[Object Detection on Desktop](./object_detection_desktop.md) shows how to run
object detection models (TensorFlow and TFLite) using the MediaPipe C++ APIs.

[Sobel edge detection]:https://en.wikipedia.org/wiki/Sobel_operator
[CameraX]:https://developer.android.com/training/camerax
