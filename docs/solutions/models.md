---
layout: default
title: Models and Model Cards
parent: Solutions
nav_order: 30
---

# MediaPipe Models and Model Cards
{: .no_toc }

1. TOC
{:toc}
---

### [Face Detection](https://google.github.io/mediapipe/solutions/face_detection)

*   Face detection model for front-facing/selfie camera:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite),
    [TFLite model quantized for EdgeTPU/Coral](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/models/face-detector-quantized_edgetpu.tflite)
*   Face detection model for back-facing camera:
    [TFLite model ](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_back.tflite)
*   [Model card](https://mediapipe.page.link/blazeface-mc)

### [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)

*   Face landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/facemesh/1)
*   [Model card](https://mediapipe.page.link/facemesh-mc)

### [Iris](https://google.github.io/mediapipe/solutions/iris)

*   Iris landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/iris_landmark/iris_landmark.tflite)
*   [Model card](https://mediapipe.page.link/iris-mc)

### [Hands](https://google.github.io/mediapipe/solutions/hands)

*   Palm detection model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/palm_detection/palm_detection.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/handdetector/1)
*   Hand landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/hand_landmark/hand_landmark.tflite),
    [TFLite model (sparse)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/hand_landmark/hand_landmark_sparse.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/handskeleton/1)
*   [Model card](https://mediapipe.page.link/handmc), [Model card (sparse)](https://mediapipe.page.link/handmc-sparse)

### [Pose](https://google.github.io/mediapipe/solutions/pose)

*   Pose detection model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_detection/pose_detection.tflite)
*   Full-body pose landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_full_body.tflite)
*   Upper-body pose landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body.tflite)
*   [Model card](https://mediapipe.page.link/blazepose-mc)

### [Holistic](https://google.github.io/mediapipe/solutions/holistic)

*   Hand recrop model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/holistic_landmark/hand_recrop.tflite)

### [Hair Segmentation](https://google.github.io/mediapipe/solutions/hair_segmentation)

*   [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hair_segmentation.tflite)
*   [Model card](https://mediapipe.page.link/hairsegmentation-mc)

### [Object Detection](https://google.github.io/mediapipe/solutions/object_detection)

*   [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/ssdlite_object_detection.tflite)
*   [TFLite model quantized for EdgeTPU/Coral](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/models/object-detector-quantized_edgetpu.tflite)
*   [TensorFlow model](https://github.com/google/mediapipe/tree/master/mediapipe/models/object_detection_saved_model)
*   [Model information](https://github.com/google/mediapipe/tree/master/mediapipe/models/object_detection_saved_model/README.md)

### [Objectron](https://google.github.io/mediapipe/solutions/objectron)

*   [TFLite model for shoes](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/object_detection_3d_sneakers.tflite)
*   [TFLite model for chairs](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/object_detection_3d_chair.tflite)
*   [TFLite model for cameras](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/object_detection_3d_camera.tflite)
*   [TFLite model for cups](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/object_detection_3d_cup.tflite)
*   [Single-stage TFLite model for shoes](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/object_detection_3d_sneakers_1stage.tflite)
*   [Single-stage TFLite model for chairs](https://github.com/google/mediapipe/tree/master/mediapipe/modules/objectron/object_detection_3d_chair_1stage.tflite)
*   [Model card](https://mediapipe.page.link/objectron-mc)

### [KNIFT](https://google.github.io/mediapipe/solutions/knift)

*   [TFLite model for up to 200 keypoints](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_float.tflite)
*   [TFLite model for up to 400 keypoints](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_float_400.tflite)
*   [TFLite model for up to 1000 keypoints](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_float_1k.tflite)
*   [Model card](https://mediapipe.page.link/knift-mc)
