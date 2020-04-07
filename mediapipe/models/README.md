## MediaPipe Models

Here are descriptions of the models used in the [example applications](../docs/examples.md).

### Object Detection
  * [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/ssdlite_object_detection.tflite)
  * See [here](object_detection_saved_model/README.md) for model details.

### Face Detection
  * [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite)
  * Paper: ["BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"](https://arxiv.org/abs/1907.05047)
  * [Model card](https://sites.google.com/corp/view/perception-cv4arvr/blazeface#h.p_21ojPZDx3cqq)

### Face Mesh
  * [TF.js model](https://tfhub.dev/mediapipe/facemesh/1)
  * Paper: ["Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs"](https://arxiv.org/abs/1907.06724)
  * [TensorFlow Blog post](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)
  * [Model card](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view)

### Hand Detection and Tracking
  * Palm detection: [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/palm_detection.tflite), [TF.js model](https://tfhub.dev/mediapipe/handdetector/1)
  * 2D hand landmark: [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark.tflite)
  * 3D hand landmark: [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark_3d.tflite), [TF.js model](https://tfhub.dev/mediapipe/handskeleton/1)
  * [Google AI Blog post](https://mediapipe.page.link/handgoogleaiblog)
  * [TensorFlow Blog post](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)
  * [Model card](https://mediapipe.page.link/handmc)

### Hair Segmentation
  * [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hair_segmentation.tflite)
  * Paper: ["Real-time Hair segmentation and recoloring on Mobile GPUs"](https://arxiv.org/abs/1907.06740)
  * [Model card](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation#h.p_NimuO7PgHxlY)
