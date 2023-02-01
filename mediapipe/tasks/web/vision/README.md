# MediaPipe Tasks Vision Package

This package contains the vision tasks for MediaPipe.

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
    "model.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
imageSegmenter.segment(image, (masks, width, height) => {
  ...
});
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

## Handlandmark Detection

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

