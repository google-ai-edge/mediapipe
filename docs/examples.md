---
nav_exclude: true
---

# Examples

Below are code samples on how to run MediaPipe on both mobile and desktop. We
currently support MediaPipe APIs on mobile for Android only but will add support
for Objective-C shortly.

## Mobile

### [Hello World! on Android](./getting_started/hello_world_android.md)

This should be the first mobile Android example users go through in detail. It
teaches the following:

*   Introduction of a simple MediaPipe graph running on mobile GPUs for
    [Sobel edge detection](https://en.wikipedia.org/wiki/Sobel_operator).
*   Building a simple baseline Android application that displays "Hello World!".
*   Adding camera preview support into the baseline application using the
    Android [CameraX] API.
*   Incorporating the Sobel edge detection graph to process the live camera
    preview and display the processed video in real-time.

[Sobel edge detection]:https://en.wikipedia.org/wiki/Sobel_operator
[CameraX]:https://developer.android.com/training/camerax

### [Hello World! on iOS](./getting_started/hello_world_ios.md)

This is the iOS version of Sobel edge detection example.

### [Face Detection](./solutions/face_detection.md)

### [Face Mesh](./solutions/face_mesh.md)

### [Hand](./solutions/hand.md)

### [Hair Segmentation](./solutions/hair_segmentation.md)

### [Object Detection](./solutions/object_detection.md)

### [Box Tracking](./solutions/box_tracking.md)

### [Objectron: 3D Object Detection](./solutions/objectron.md)

### [KNIFT: Template-based Feature Matching](./solutions/knift.md)

## Desktop

### [Hello World for C++](./getting_started/hello_world_desktop.md)

This shows how to run a simple graph using the MediaPipe C++ APIs.

### [Face Detection](./solutions/face_detection.md)

### [Face Mesh](./solutions/face_mesh.md)

### [Hand](./solutions/hand.md)

### [Hair Segmentation](./solutions/hair_segmentation.md)

### [Object Detection](./solutions/object_detection.md)

### [Box Tracking](./solutions/box_tracking.md)

### [AutoFlip - Semantic-aware Video Cropping](./solutions/autoflip.md)

### [Preparing Data Sets with MediaSequence](./solutions/media_sequence.md)

This shows how to use MediaPipe for media processing to prepare video data sets
for training a TensorFlow model.

### [Feature Extraction and Model Inference for YouTube-8M Challenge](./solutions/youtube_8m.md)

This shows how to use MediaPipe to prepare training data for the YouTube-8M
Challenge and do the model inference with the baseline model.

## Google Coral (ML acceleration with Google EdgeTPU)

### [Face Detection](./solutions/face_detection.md)

### [Object Detection](./solutions/object_detection.md)

## Web Browser

See more details [here](./getting_started/web.md) and
[Google Developer blog post](https://mediapipe.page.link/webdevblog).

### [Face Detection in Browser](https://viz.mediapipe.dev/demo/face_detection)

### [Hand Detection in Browser](https://viz.mediapipe.dev/demo/hand_detection)

### [Hand Tracking in Browser](https://viz.mediapipe.dev/demo/hand_tracking)

### [Hair Segmentation in Browser](https://viz.mediapipe.dev/demo/hair_segmentation)
