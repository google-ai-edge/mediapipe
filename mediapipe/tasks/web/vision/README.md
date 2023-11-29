# MediaPipe Tasks Vision Package

This package contains the vision tasks for MediaPipe.

## Face Detector

The MediaPipe Face Detector task lets you detect the presence and location of
faces within images or videos.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const faceDetector = await FaceDetector.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const detections = faceDetector.detect(image);
```

For more information, refer to the [Face Detector](https://developers.google.com/mediapipe/solutions/vision/face_detector/web_js) documentation.

## Face Landmarker

The MediaPipe Face Landmarker task lets you detect the landmarks of faces in
an image. You can use this Task to localize key points of a face and render
visual effects over the faces.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const faceLandmarker = await FaceLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = faceLandmarker.detect(image);
```

For more information, refer to the [Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/web_js) documentation.

## Face Stylizer

The MediaPipe Face Stylizer lets you perform face stylization on images.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const faceStylizer = await FaceStylizer.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/1/blaze_face_stylizer.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const stylizedImage = faceStylizer.stylize(image);
```

## Gesture Recognizer

The MediaPipe Gesture Recognizer task lets you recognize hand gestures in real
time, and provides the recognized hand gesture results along with the landmarks
of the detected hands. You can use this task to recognize specific hand gestures
from a user, and invoke application features that correspond to those gestures.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const gestureRecognizer = await GestureRecognizer.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const recognitions = gestureRecognizer.recognize(image);
```

For more information, refer to the [Gesture Recognizer](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/web_js) documentation.

## Hand Landmarker

The MediaPipe Hand Landmarker task lets you detect the landmarks of the hands in
an image. You can use this Task to localize key points of the hands and render
visual effects over the hands.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const handLandmarker = await HandLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = handLandmarker.detect(image);
```

For more information, refer to the [Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/web_js) documentation.

## Image Classifier

The MediaPipe Image Classifier task lets you perform classification on images.
You can use this task to identify what an image represents among a set of
categories defined at training time.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const imageClassifier = await ImageClassifier.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const classifications = imageClassifier.classify(image);
```

For more information, refer to the [Image Classifier](https://developers.google.com/mediapipe/solutions/vision/image_classifier/web_js) documentation.

## Image Embedder

The MediaPipe Image Embedder extracts embeddings from an image.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const imageEmbedder = await ImageEmbedder.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const embeddings = imageSegmenter.embed(image);
```

## Image Segmenter

The MediaPipe Image Segmenter lets you segment an image into categories.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const imageSegmenter = await ImageSegmenter.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
imageSegmenter.segment(image, (masks, width, height) => {
  ...
});
```

For more information, refer to the [Image Segmenter](https://developers.google.com/mediapipe/solutions/vision/image_segmenter/web_js) documentation.

## Interactive Segmenter

The MediaPipe Interactive Segmenter lets you select a region of interest to
segment an image by.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const interactiveSegmenter = await InteractiveSegmenter.createFromModelPath(
    vision,
    "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
interactiveSegmenter.segment(image, { keypoint: { x: 0.1, y: 0.2 } },
    (masks, width, height) => { ... }
);
```

For more information, refer to the [Interactive Segmenter](https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter/web_js) documentation.

## Object Detector

The MediaPipe Object Detector task lets you detect the presence and location of
multiple classes of objects within images or videos.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const objectDetector = await ObjectDetector.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const detections = objectDetector.detect(image);
```

For more information, refer to the [Object Detector](https://developers.google.com/mediapipe/solutions/vision/object_detector/web_js) documentation.

## Pose Landmarker

The MediaPipe Pose Landmarker task lets you detect the landmarks of body poses
in an image. You can use this Task to localize key points of a pose and render
visual effects over the body.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const poseLandmarker = await PoseLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = poseLandmarker.detect(image);
```

For more information, refer to the [Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/web_js) documentation.
