# Examples

Below are code samples on how to run MediaPipe on both mobile and desktop. We
currently support MediaPipe APIs on mobile for Android only but will add support
for Objective-C shortly.

## Mobile

### Hello World! on Android

[Hello World! on Android](./hello_world_android.md) should be the first mobile
example users go through in detail. It teaches the following:

*   Introduction of a simple MediaPipe graph running on mobile GPUs for
    [Sobel edge detection].
*   Building a simple baseline Android application that displays "Hello World!".
*   Adding camera preview support into the baseline application using the
    Android [CameraX] API.
*   Incorporating the Sobel edge detection graph to process the live camera
    preview and display the processed video in real-time.

### Object Detection with GPU on Android

[Object Detection on GPU on Android](./object_detection_android_gpu.md)
illustrates how to use MediaPipe with a TFLite model for object detection in a
GPU-accelerated pipeline.

### Object Detection with CPU on Android

[Object Detection on CPU on Android](./object_detection_android_cpu.md)
illustrates using the same TFLite model in a CPU-based pipeline. This example
highlights how graphs can be easily adapted to run on CPU v.s. GPU.

### Face Detection on Android

[Face Detection on Android](./face_detection_android_gpu.md) illustrates how to
use MediaPipe with a TFLite model for face detection in a GPU-accelerated
pipeline.

*   The selfie face detection TFLite model is based on
    ["BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"](https://sites.google.com/view/perception-cv4arvr/blazeface).
*   [Model card](https://sites.google.com/corp/view/perception-cv4arvr/blazeface#h.p_21ojPZDx3cqq).

### Hair Segmentation on Android

[Hair Segmentation on Android](./hair_segmentation_android_gpu.md) illustrates
how to use MediaPipe with a TFLite model for hair segmentation in a
GPU-accelerated pipeline.

*   The selfie hair segmentation TFLite model is based on
    ["Real-time Hair segmentation and recoloring on Mobile GPUs"](https://sites.google.com/view/perception-cv4arvr/hair-segmentation).
*   [Model card](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation#h.p_NimuO7PgHxlY).

## Desktop

### Hello World for C++

[Hello World for C++](./hello_world_desktop.md) shows how to run a simple graph
using the MediaPipe C++ APIs.

### Preparing Data Sets with MediaSequence

[Preparing Data Sets with MediaSequence](./media_sequence.md) shows how to use
MediaPipe for media processing to prepare video data sets for training a
TensorFlow model.

### Object Detection on Desktop

[Object Detection on Desktop](./object_detection_desktop.md) shows how to run
object detection models (TensorFlow and TFLite) using the MediaPipe C++ APIs.

[Sobel edge detection]:https://en.wikipedia.org/wiki/Sobel_operator
[CameraX]:https://developer.android.com/training/camerax
