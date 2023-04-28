# MediaPipe Tasks Vision Package

This package contains the vision tasks for MediaPipe.

## Face Detection

The MediaPipe Face Detector task lets you detect the presence and location of
faces within images or videos.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const faceDetector = await FaceDetector.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/face_detector/face_detection_short_range.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const detections = faceDetector.detect(image);
```

## Face Landmark Detection

The MediaPipe Face Landmarker task lets you detect the landmarks of faces in
an image. You can use this Task to localize key points of a face and render
visual effects over the faces.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const faceLandmarker = await FaceLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = faceLandmarker.detect(image);
```

## Face Stylizer

The MediaPipe Face Stylizer lets you perform face stylization on images.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const faceStylizer = await FaceStylizer.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/face_stylizer/face_stylizer_with_metadata.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const stylizedImage = faceStylizer.stylize(image);
```

## Gesture Recognition

The MediaPipe Gesture Recognizer task lets you recognize hand gestures in real
time, and provides the recognized hand gesture results along with the landmarks
of the detected hands. You can use this task to recognize specific hand gestures
from a user, and invoke application features that correspond to those gestures.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const gestureRecognizer = await GestureRecognizer.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const recognitions = gestureRecognizer.recognize(image);
```

## Hand Landmark Detection

The MediaPipe Hand Landmarker task lets you detect the landmarks of the hands in
an image. You can use this Task to localize key points of the hands and render
visual effects over the hands.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const handLandmarker = await HandLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = handLandmarker.detect(image);
```

For more information, refer to the [Handlandmark Detection](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/web_js) documentation.

## Image Classification

The MediaPipe Image Classifier task lets you perform classification on images.
You can use this task to identify what an image represents among a set of
categories defined at training time.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const imageClassifier = await ImageClassifier.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/image_classifier/efficientnet_lite0_uint8.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const classifications = imageClassifier.classify(image);
```

For more information, refer to the [Image Classification](https://developers.google.com/mediapipe/solutions/vision/image_classifier/web_js) documentation.

## Image Segmentation

The MediaPipe Image Segmenter lets you segment an image into categories.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const imageSegmenter = await ImageSegmenter.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/image_segmenter/selfie_segmentation.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
imageSegmenter.segment(image, (masks, width, height) => {
  ...
});
```

## Interactive Segmentation

The MediaPipe Interactive Segmenter lets you select a region of interest to
segment an image by.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const interactiveSegmenter = await InteractiveSegmenter.createFromModelPath(
    vision,
    "https://storage.googleapis.com/mediapipe-tasks/interactive_segmenter/ptm_512_hdt_ptm_woid.tflite
);
const image = document.getElementById("image") as HTMLImageElement;
interactiveSegmenter.segment(image, { keypoint: { x: 0.1, y: 0.2 } },
    (masks, width, height) => { ... }
);
```

## Object Detection

The MediaPipe Object Detector task lets you detect the presence and location of
multiple classes of objects within images or videos.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const objectDetector = await ObjectDetector.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const detections = objectDetector.detect(image);
```

For more information, refer to the [Object Detector](https://developers.google.com/mediapipe/solutions/vision/object_detector/web_js) documentation.


## Pose Landmark Detection

The MediaPipe Pose Landmarker task lets you detect the landmarks of body poses
in an image. You can use this Task to localize key points of a pose and render
visual effects over the body.

```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const poseLandmarker = await PoseLandmarker.createFromModelPath(vision,
    "model.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = poseLandmarker.detect(image);
```
