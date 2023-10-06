---
layout: forward
target: https://developers.google.com/mediapipe/solutions/vision/face_detector/
title: Face Detection
parent: MediaPipe Legacy Solutions
nav_order: 1
---

# MediaPipe Face Detection
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
As of May 10, 2023, this solution was upgraded to a new MediaPipe
Solution. For more information, see the
[MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/face_detector)
site.*

----

## Overview

MediaPipe Face Detection is an ultrafast face detection solution that comes with
6 landmarks and multi-face support. It is based on
[BlazeFace](https://arxiv.org/abs/1907.05047), a lightweight and well-performing
face detector tailored for mobile GPU inference. The detector's super-realtime
performance enables it to be applied to any live viewfinder experience that
requires an accurate facial region of interest as an input for other
task-specific models, such as 3D facial keypoint estimation (e.g.,
[MediaPipe Face Mesh](./face_mesh.md)), facial features or expression
classification, and face region segmentation. BlazeFace uses a lightweight
feature extraction network inspired by, but distinct from
[MobileNetV1/V2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html),
a GPU-friendly anchor scheme modified from
[Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325), and an
improved tie resolution strategy alternative to non-maximum suppression. For
more information about BlazeFace, please see the [Resources](#resources)
section.

![face_detection_android_gpu.gif](https://mediapipe.dev/images/mobile/face_detection_android_gpu.gif)

## Solution APIs

### Configuration Options

Naming style and availability may differ slightly across platforms/languages.

#### model_selection

An integer index `0` or `1`. Use `0` to select a short-range model that works
best for faces within 2 meters from the camera, and `1` for a full-range model
best for faces within 5 meters. For the full-range option, a sparse model is
used for its improved inference speed. Please refer to the
[model cards](./models.md#face_detection) for details. Default to `0` if not
specified.

Note: Not available for JavaScript (use "model" instead).

#### model

A string value to indicate which model should be used. Use "short" to
select a short-range model that works best for faces within 2 meters from the
camera, and "full" for a full-range model best for faces within 5 meters. For
the full-range option, a sparse model is used for its improved inference speed.
Please refer to the model cards for details. Default to empty string.

Note: Valid only for JavaScript solution.

#### selfie_mode

A boolean value to indicate whether to flip the images/video frames
horizontally or not. Default to `false`.

Note: Valid only for JavaScript solution.

#### min_detection_confidence

Minimum confidence value (`[0.0, 1.0]`) from the face detection model for the
detection to be considered successful. Default to `0.5`.

### Output

Naming style may differ slightly across platforms/languages.

#### detections

Collection of detected faces, where each face is represented as a detection
proto message that contains a bounding box and 6 key points (right eye, left
eye, nose tip, mouth center, right ear tragion, and left ear tragion). The
bounding box is composed of `xmin` and `width` (both normalized to `[0.0, 1.0]`
by the image width) and `ymin` and `height` (both normalized to `[0.0, 1.0]` by
the image height). Each key point is composed of `x` and `y`, which are
normalized to `[0.0, 1.0]` by the image width and height respectively.

### Python Solution API

Please first follow general [instructions](../getting_started/python.md) to
install MediaPipe Python package, then learn more in the companion
[Python Colab](#resources) and the usage example below.

Supported configuration options:

*   [model_selection](#model_selection)
*   [min_detection_confidence](#min_detection_confidence)

```python
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

### JavaScript Solution API

Please first see general [introduction](../getting_started/javascript.md) on
MediaPipe in JavaScript, then learn more in the companion [web demo](#resources)
and the following usage example.

Supported face detection options:
*   [selfieMode](#selfie_mode)
*   [model](#model)
*   [minDetectionConfidence](#min_detection_confidence)

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
  </div>
</body>
</html>
```

```javascript
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const drawingUtils = window;

function onResults(results) {
  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.detections.length > 0) {
    drawingUtils.drawRectangle(
        canvasCtx, results.detections[0].boundingBox,
        {color: 'blue', lineWidth: 4, fillColor: '#00000000'});
    drawingUtils.drawLandmarks(canvasCtx, results.detections[0].landmarks, {
      color: 'red',
      radius: 5,
    });
  }
  canvasCtx.restore();
}

const faceDetection = new FaceDetection({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.0/${file}`;
}});
faceDetection.setOptions({
  model: 'short',
  minDetectionConfidence: 0.5
});
faceDetection.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceDetection.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

### Android Solution API

Please first follow general
[instructions](../getting_started/android_solutions.md) to add MediaPipe Gradle
dependencies and try the Android Solution API in the companion
[example Android Studio project](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/solutions/facedetection),
and learn more in the usage example below.

Supported configuration options:

*   [staticImageMode](#static_image_mode)
*   [modelSelection](#model_selection)

#### Camera Input

```java
// For camera input and result rendering with OpenGL.
FaceDetectionOptions faceDetectionOptions =
    FaceDetectionOptions.builder()
        .setStaticImageMode(false)
        .setModelSelection(0).build();
FaceDetection faceDetection = new FaceDetection(this, faceDetectionOptions);
faceDetection.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));

// Initializes a new CameraInput instance and connects it to MediaPipe Face Detection Solution.
CameraInput cameraInput = new CameraInput(this);
cameraInput.setNewFrameListener(
    textureFrame -> faceDetection.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<FaceDetectionResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/facedetection/src/main/java/com/google/mediapipe/examples/facedetection/FaceDetectionResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<FaceDetectionResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, faceDetection.getGlContext(), faceDetection.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new FaceDetectionResultGlRenderer());
glSurfaceView.setRenderInputImage(true);
faceDetection.setResultListener(
    faceDetectionResult -> {
      if (faceDetectionResult.multiFaceDetections().isEmpty()) {
        return;
      }
      RelativeKeypoint noseTip =
          faceDetectionResult
              .multiFaceDetections()
              .get(0)
              .getLocationData()
              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Detection nose tip normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseTip.getX(), noseTip.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(faceDetectionResult);
      glSurfaceView.requestRender();
    });

// The runnable to start camera after the GLSurfaceView is attached.
glSurfaceView.post(
    () ->
        cameraInput.start(
            this,
            faceDetection.getGlContext(),
            CameraInput.CameraFacing.FRONT,
            glSurfaceView.getWidth(),
            glSurfaceView.getHeight()));
```

#### Image Input

```java
// For reading images from gallery and drawing the output in an ImageView.
FaceDetectionOptions faceDetectionOptions =
    FaceDetectionOptions.builder()
        .setStaticImageMode(true)
        .setModelSelection(0).build();
FaceDetection faceDetection = new FaceDetection(this, faceDetectionOptions);

// Connects MediaPipe Face Detection Solution to the user-defined ImageView
// instance that allows users to have the custom drawing of the output landmarks
// on it. See mediapipe/examples/android/solutions/facedetection/src/main/java/com/google/mediapipe/examples/facedetection/FaceDetectionResultImageView.java
// as an example.
FaceDetectionResultImageView imageView = new FaceDetectionResultImageView(this);
faceDetection.setResultListener(
    faceDetectionResult -> {
      if (faceDetectionResult.multiFaceDetections().isEmpty()) {
        return;
      }
      int width = faceDetectionResult.inputBitmap().getWidth();
      int height = faceDetectionResult.inputBitmap().getHeight();
      RelativeKeypoint noseTip =
          faceDetectionResult
              .multiFaceDetections()
              .get(0)
              .getLocationData()
              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Detection nose tip coordinates (pixel values): x=%f, y=%f",
              noseTip.getX() * width, noseTip.getY() * height));
      // Request canvas drawing.
      imageView.setFaceDetectionResult(faceDetectionResult);
      runOnUiThread(() -> imageView.update());
    });
faceDetection.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));

// ActivityResultLauncher to get an image from the gallery as Bitmap.
ActivityResultLauncher<Intent> imageGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null && result.getResultCode() == RESULT_OK) {
            Bitmap bitmap = null;
            try {
              bitmap =
                  MediaStore.Images.Media.getBitmap(
                      this.getContentResolver(), resultIntent.getData());
              // Please also rotate the Bitmap based on its orientation.
            } catch (IOException e) {
              Log.e(TAG, "Bitmap reading error:" + e);
            }
            if (bitmap != null) {
              faceDetection.send(bitmap);
            }
          }
        });
Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
imageGetter.launch(pickImageIntent);
```

#### Video Input

```java
// For video input and result rendering with OpenGL.
FaceDetectionOptions faceDetectionOptions =
    FaceDetectionOptions.builder()
        .setStaticImageMode(false)
        .setModelSelection(0).build();
FaceDetection faceDetection = new FaceDetection(this, faceDetectionOptions);
faceDetection.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));

// Initializes a new VideoInput instance and connects it to MediaPipe Face Detection Solution.
VideoInput videoInput = new VideoInput(this);
videoInput.setNewFrameListener(
    textureFrame -> faceDetection.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<FaceDetectionResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/facedetection/src/main/java/com/google/mediapipe/examples/facedetection/FaceDetectionResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<FaceDetectionResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, faceDetection.getGlContext(), faceDetection.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new FaceDetectionResultGlRenderer());
glSurfaceView.setRenderInputImage(true);

faceDetection.setResultListener(
    faceDetectionResult -> {
      if (faceDetectionResult.multiFaceDetections().isEmpty()) {
        return;
      }
      RelativeKeypoint noseTip =
          faceDetectionResult
              .multiFaceDetections()
              .get(0)
              .getLocationData()
              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Detection nose tip normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseTip.getX(), noseTip.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(faceDetectionResult);
      glSurfaceView.requestRender();
    });

ActivityResultLauncher<Intent> videoGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null) {
            if (result.getResultCode() == RESULT_OK) {
              glSurfaceView.post(
                  () ->
                      videoInput.start(
                          this,
                          resultIntent.getData(),
                          faceDetection.getGlContext(),
                          glSurfaceView.getWidth(),
                          glSurfaceView.getHeight()));
            }
          }
        });
Intent pickVideoIntent = new Intent(Intent.ACTION_PICK);
pickVideoIntent.setDataAndType(MediaStore.Video.Media.INTERNAL_CONTENT_URI, "video/*");
videoGetter.launch(pickVideoIntent);
```

## Example Apps

Please first see general instructions for
[Android](../getting_started/android.md), [iOS](../getting_started/ios.md) and
[desktop](../getting_started/cpp.md) on how to build MediaPipe examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Mobile

#### GPU Pipeline

*   Graph:
    [`mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1DZTCy1gp238kkMnu4fUkwI3IrF77Mhy5)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectiongpu:facedetectiongpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectiongpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/facedetectiongpu:FaceDetectionGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/facedetectiongpu/BUILD)

#### CPU Pipeline

This is very similar to the [GPU pipeline](#gpu-pipeline) except that at the
beginning and the end of the pipeline it performs GPU-to-CPU and CPU-to-GPU
image transfer respectively. As a result, the rest of graph, which shares the
same configuration as the GPU pipeline, runs entirely on CPU.

*   Graph:
    [`mediapipe/graphs/face_detection/face_detection_mobile_cpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_cpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1npiZY47jbO5m2YaL63o5QoCQs40JC6C7)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu:facedetectioncpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/facedetectioncpu:FaceDetectionCpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/facedetectioncpu/BUILD)

### Desktop

*   Running on CPU:
    *   Graph:
        [`mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/face_detection:face_detection_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/face_detection/BUILD)
*   Running on GPU
    *   Graph:
        [`mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/face_detection:face_detection_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/face_detection/BUILD)

### Coral

Please refer to
[these instructions](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/README.md)
to cross-compile and run MediaPipe examples on the
[Coral Dev Board](https://coral.ai/products/dev-board).

## Resources

*   Paper:
    [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/abs/1907.05047)
    ([presentation](https://docs.google.com/presentation/d/1YCtASfnYyZtH-41QvnW5iZxELFnf0MF-pPWSLGj8yjQ/present?slide=id.g5bc8aeffdd_1_0))
    ([poster](https://drive.google.com/file/d/1u6aB6wxDY7X2TmeUUKgFydulNtXkb3pu/view))
*   [Models and model cards](./models.md#face_detection)
*   [Web demo](https://code.mediapipe.dev/codepen/face_detection)
*   [Python Colab](https://mediapipe.page.link/face_detection_py_colab)
