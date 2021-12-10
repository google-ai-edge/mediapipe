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

*   Short-range model (best for faces within 2 meters from the camera):
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection/face_detection_short_range.tflite),
    [TFLite model quantized for EdgeTPU/Coral](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/models/face-detector-quantized_edgetpu.tflite),
    [Model card](https://mediapipe.page.link/blazeface-mc)
*   Full-range model (dense, best for faces within 5 meters from the camera):
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection/face_detection_full_range.tflite),
    [Model card](https://mediapipe.page.link/blazeface-back-mc)
*   Full-range model (sparse, best for faces within 5 meters from the camera):
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite),
    [Model card](https://mediapipe.page.link/blazeface-back-sparse-mc)

Full-range dense and sparse models have the same quality in terms of
[F-score](https://en.wikipedia.org/wiki/F-score) however differ in underlying
metrics. The dense model is slightly better in
[Recall](https://en.wikipedia.org/wiki/Precision_and_recall) whereas the sparse
model outperforms the dense one in
[Precision](https://en.wikipedia.org/wiki/Precision_and_recall). Speed-wise
sparse model is ~30% faster when executing on CPU via
[XNNPACK](https://github.com/google/XNNPACK) whereas on GPU the models
demonstrate comparable latencies. Depending on your application, you may prefer
one over the other.

### [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)

*   Face landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/facemesh/1)
*   Face landmark model w/ attention (aka Attention Mesh):
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark_with_attention.tflite)
*   [Model card](https://mediapipe.page.link/facemesh-mc),
    [Model card (w/ attention)](https://mediapipe.page.link/attentionmesh-mc)

### [Iris](https://google.github.io/mediapipe/solutions/iris)

*   Iris landmark model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/iris_landmark/iris_landmark.tflite)
*   [Model card](https://mediapipe.page.link/iris-mc)

### [Hands](https://google.github.io/mediapipe/solutions/hands)

*   Palm detection model:
    [TFLite model (lite)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/palm_detection/palm_detection_lite.tflite),
    [TFLite model (full)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/palm_detection/palm_detection_full.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/handdetector/1)
*   Hand landmark model:
    [TFLite model (lite)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite),
    [TFLite model (full)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/hand_landmark/hand_landmark_full.tflite),
    [TF.js model](https://tfhub.dev/mediapipe/handskeleton/1)
*   [Model card](https://mediapipe.page.link/handmc)

### [Pose](https://google.github.io/mediapipe/solutions/pose)

*   Pose detection model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_detection/pose_detection.tflite)
*   Pose landmark model:
    [TFLite model (lite)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite),
    [TFLite model (full)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_full.tflite),
    [TFLite model (heavy)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite)
*   [Model card](https://mediapipe.page.link/blazepose-mc)

### [Holistic](https://google.github.io/mediapipe/solutions/holistic)

*   Hand recrop model:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/modules/holistic_landmark/hand_recrop.tflite)

### [Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation)

*   [TFLite model (general)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/selfie_segmentation/selfie_segmentation.tflite)
*   [TFLite model (landscape)](https://github.com/google/mediapipe/tree/master/mediapipe/modules/selfie_segmentation/selfie_segmentation_landscape.tflite)
*   [Model card](https://mediapipe.page.link/selfiesegmentation-mc)

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
