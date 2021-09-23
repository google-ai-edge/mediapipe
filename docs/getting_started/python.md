---
layout: default
title: MediaPipe in Python
parent: Getting Started
has_children: true
has_toc: false
nav_order: 3
---

# MediaPipe in Python
{: .no_toc }

1. TOC
{:toc}
---

## Ready-to-use Python Solutions

MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt
Python package. MediaPipe Python package is available on
[PyPI](https://pypi.org/project/mediapipe/) for Linux, macOS and Windows.

You can, for instance, activate a Python virtual environment:

```bash
$ python3 -m venv mp_env && source mp_env/bin/activate
```

Install MediaPipe Python package and start Python interpreter:

```bash
(mp_env)$ pip install mediapipe
(mp_env)$ python3
```

In Python interpreter, import the package and start using one of the solutions:

```python
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
```

Tip: Use command `deactivate` to later exit the Python virtual environment.

To learn more about configuration options and usage examples, please find
details in each solution via the links below:

*   [MediaPipe Face Detection](../solutions/face_detection#python-solution-api)
*   [MediaPipe Face Mesh](../solutions/face_mesh#python-solution-api)
*   [MediaPipe Hands](../solutions/hands#python-solution-api)
*   [MediaPipe Holistic](../solutions/holistic#python-solution-api)
*   [MediaPipe Objectron](../solutions/objectron#python-solution-api)
*   [MediaPipe Pose](../solutions/pose#python-solution-api)
*   [MediaPipe Selfie Segmentation](../solutions/selfie_segmentation#python-solution-api)

## MediaPipe on Google Colab

*   [MediaPipe Face Detection Colab](https://mediapipe.page.link/face_detection_py_colab)
*   [MediaPipe Face Mesh Colab](https://mediapipe.page.link/face_mesh_py_colab)
*   [MediaPipe Hands Colab](https://mediapipe.page.link/hands_py_colab)
*   [MediaPipe Holistic Colab](https://mediapipe.page.link/holistic_py_colab)
*   [MediaPipe Objectron Colab](https://mediapipe.page.link/objectron_py_colab)
*   [MediaPipe Pose Colab](https://mediapipe.page.link/pose_py_colab)
*   [MediaPipe Pose Classification Colab (Basic)](https://mediapipe.page.link/pose_classification_basic)
*   [MediaPipe Pose Classification Colab (Extended)](https://mediapipe.page.link/pose_classification_extended)
*   [MediaPipe Selfie Segmentation Colab](https://mediapipe.page.link/selfie_segmentation_py_colab)

## MediaPipe Python Framework

The ready-to-use solutions are built upon the MediaPipe Python framework, which
can be used by advanced users to run their own MediaPipe graphs in Python.
Please see [here](./python_framework.md) for more info.

## Building MediaPipe Python Package

Follow the steps below only if you have local changes and need to build the
Python package from source. Otherwise, we strongly encourage our users to simply
run `pip install mediapipe` to use the ready-to-use solutions, more convenient
and much faster.

MediaPipe PyPI currently doesn't provide aarch64 Python wheel
files. For building and using MediaPipe Python on aarch64 Linux systems such as
Nvidia Jetson and Raspberry Pi, please read
[here](https://github.com/jiuqiant/mediapipe-python-aarch64).

1.  Make sure that Bazel and OpenCV are correctly installed and configured for
    MediaPipe. Please see [Installation](./install.md) for how to setup Bazel
    and OpenCV for MediaPipe on Linux and macOS.

2.  Install the following dependencies.

    Debian or Ubuntu:

    ```bash
    $ sudo apt install python3-dev
    $ sudo apt install python3-venv
    $ sudo apt install -y protobuf-compiler

    # If you need to build opencv from source.
    $ sudo apt install cmake
    ```

    macOS:

    ```bash
    $ brew install protobuf

    # If you need to build opencv from source.
    $ brew install cmake
    ```

    Windows:

    Download the latest protoc win64 zip from
    [the Protobuf GitHub repo](https://github.com/protocolbuffers/protobuf/releases),
    unzip the file, and copy the protoc.exe executable to a preferred
    location. Please ensure that location is added into the Path environment
    variable.

3.  Activate a Python virtual environment.

    ```bash
    $ python3 -m venv mp_env && source mp_env/bin/activate
    ```

4.  In the virtual environment, go to the MediaPipe repo directory.

5.  Install the required Python packages.

    ```bash
    (mp_env)mediapipe$ pip3 install -r requirements.txt
    ```

6.  Generate and install MediaPipe package.

    ```bash
    (mp_env)mediapipe$ python3 setup.py gen_protos
    (mp_env)mediapipe$ python3 setup.py install --link-opencv
    ```

    or

    ```bash
    (mp_env)mediapipe$ python3 setup.py gen_protos
    (mp_env)mediapipe$ python3 setup.py bdist_wheel
    ```
