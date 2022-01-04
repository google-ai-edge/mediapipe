## Description
The pose_tracking_dll module allows for building a Mediapipe-based pose tracking DLL library that can be used with any C++ project. All the dependencies such as tensorflow are built statically into the dll. 

Currently, the following features are supported:
- Segmenting the person(s) of interest
- Segmenting the skeleton(s)
- Accessing the 3D coordinates of each node of the skeleton

## Prerequisites
Follow the guidelines on the official Mediapipe website: https://google.github.io/mediapipe/getting_started/install.html#installing-on-windows

IMPORTANT: The tutorial does not specify which version of Bazel to install. Install Bazel version 3.7.2

## How to build
Assuming you're in the root of the repository:

cd mediapipe

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 pose_tracking_dll:pose_tracking_cpu

The results will be stored in bazel-bin\mediapipe\pose_tracking_dll folder.

## How to use
Go to bazel-bin\mediapipe\pose_tracking_dll

Link pose_tracking_cpu.lib and pose_tracking_lib.dll.if.lib statically in your project.

Make sure the opencv_world3410.dll and pose_tracking_lib.dll are accessible in your working directory.

Use mediapipe\pose_tracking_dll\pose_tracking.h header file to access the methods of the library.
