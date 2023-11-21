# OpenVINO&trade; Model Server fork of [MediaPipe](https://google.github.io/mediapipe/).
This repository allows users to take advantage of OpenVINO&trade; in the Mediapipe framework. It includes inference calculators which can replace Tensorflow backend with OpenVINO&trade; Runtime.
That way, you can expect more efficient execution and lower latency on CPU.

Check the included demos with pipeline examples or create your own graphs and execution flows.

# List of changes introduced in this repository fork

- added [Dockerfile.openvino](Dockerfile.openvino) dockerfile that creates runtime and development environment.
- added [Makefile](Makefile) file with build, test and demo targets for the ease of use.
- modified [build_desktop_examples.sh](build_desktop_examples.sh) script to build new demos.
- added [calculators](mediapipe/calculators/ovms) and [calculators](mediapipe/calculators/openvino) for OpenVINO&trade; inference in mediapipe graphs
  detailed [description](mediapipe/calculators/ovms/calculators.md).
- modified bazel [targets](mediapipe/examples/desktop) to use OpenVINO&trade; inference calculators (the list of available demos is in the table below).
- modified [WORKSPACE](WORKSPACE) file to add OpenVINO&trade; Model Server dependencies.
  Specifically target @ovms//src:ovms_lib as dependency from [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
- modified [graphs and bazel targets](mediapipe/modules/) to use OpenVINO&trade; inference instead of TensorFlow inference.
- added [setup_ovms.py](setup_ovms.py) script to create models repository used in OpenVINO&trade; inference. The script needs to be executed to prepare specific directory structures with tflite models and config.json in the [mediapipe/models/ovms](directory).
- modified setup_opecv.py to install 4.7.0 OpenCV version instead of previous 3.4.

[]() OpenVINO&trade; demo                                                    | C++                                                     | [Python](mediapipe/examples/python/README.md)                              | Original Google demo                                                        |
:---------------------------------------------------------------------------------------- | :-----------------------------------------------------: | :-----------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
[Face Detection](mediapipe/examples/desktop/face_detection/README.md)                     | ✅                                                      | ✅                                                              |[Face Detection](https://google.github.io/mediapipe/solutions/face_detection)             |
[Iris](mediapipe/examples/desktop/iris_tracking/README.md)                                | ✅                                                      |                                                                |[Iris](https://google.github.io/mediapipe/solutions/iris)                                 |
[Pose](mediapipe/examples/desktop/pose_tracking/README.md)                                | ✅                                                      |                                                                |[Pose](https://google.github.io/mediapipe/solutions/pose)                                 |
[Holistic](mediapipe/examples/desktop/holistic_tracking/README.md)                        | ✅                                                      | ✅                                                              |[Holistic](https://google.github.io/mediapipe/solutions/holistic)                         |
[Object Detection](mediapipe/examples/desktop/object_detection/README.md)                 | ✅                                                      | ✅                                                              |[Object Detection](https://google.github.io/mediapipe/solutions/object_detection)         |

# Quick start guide

Check the [quick start guide](docs/quickstartguide.md) to with easy to follow instructions for building and running the example applications and graphs.

# Development instructions

The [developer guide](docs/development.md) includes the list of instructions and practices in developing your own application and graphs.

![MediaPipe](https://mediapipe.dev/images/mediapipe_small.png)

--------------------------------------------------------------------------------

## Live ML anywhere

[MediaPipe](https://google.github.io/mediapipe/) offers cross-platform, customizable
ML solutions for live and streaming media.

![accelerated.png](https://mediapipe.dev/images/accelerated_small.png)                                                               | ![cross_platform.png](https://mediapipe.dev/images/cross_platform_small.png)
:------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------:
***End-to-End acceleration***: *Built-in fast ML inference and processing accelerated even on common hardware* | ***Build once, deploy anywhere***: *Unified solution works across Android, iOS, desktop/cloud, web and IoT*
![ready_to_use.png](https://mediapipe.dev/images/ready_to_use_small.png)                                                             | ![open_source.png](https://mediapipe.dev/images/open_source_small.png)
***Ready-to-use solutions***: *Cutting-edge ML solutions demonstrating full power of the framework*            | ***Free and open source***: *Framework and solutions both under Apache 2.0, fully extensible and customizable*

## ML solutions in MediaPipe OpenVINO&trade; fork

Face Detection                                                                                                                 | Iris                                                                                                      | Pose                                                                                                      | Holistic                                                                          | Object Detection                                              |
:----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :------: | :----------------------------------------------------------------------------------------------------------------------------------: |
[![face_detection](https://mediapipe.dev/images/mobile/face_detection_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/face_detection) | [![iris](https://mediapipe.dev/images/mobile/iris_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/iris) | [![pose](https://mediapipe.dev/images/mobile/pose_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/pose) | [![holistic_tracking](https://mediapipe.dev/images/mobile/holistic_tracking_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/holistic) | [![object_detection](https://mediapipe.dev/images/mobile/object_detection_android_gpu_small.gif)](https://google.github.io/mediapipe/solutions/object_detection) |

## Fork baseline
The fork is based on original mediapipe release origin/v0.10.3.

Original v0.10.3 Google ML solutions in MediaPipe can be found [here](https://github.com/google/mediapipe/tree/v0.10.3)