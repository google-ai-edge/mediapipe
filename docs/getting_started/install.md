---
layout: default
title: Installation
parent: Getting Started
nav_order: 1
---

# Installation
{: .no_toc }

1. TOC
{:toc}
---

Note: To interoperate with OpenCV, OpenCV 3.x and above are preferred. OpenCV
2.x currently works but interoperability support may be deprecated in the
future.

Note: If you plan to use TensorFlow calculators and example apps, there is a
known issue with gcc and g++ version 6.3 and 7.3. Please use other versions.

Note: To make Mediapipe work with TensorFlow, please set Python 3.7 as the
default Python version and install the Python "six" library by running `pip3
install --user six`.

Note: To build and run Android example apps, see these
[instructions](./building_examples.md#android). To build and run iOS example
apps, see these [instructions](./building_examples.md#ios).

## Installing on Debian and Ubuntu

1.  Checkout MediaPipe repository.

    ```bash
    $ git clone https://github.com/google/mediapipe.git

    # Change directory into MediaPipe root directory
    $ cd mediapipe
    ```

2.  Install Bazel.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-ubuntu.html)
    to install Bazel 2.0 or higher.

    For Nvidia Jetson and Raspberry Pi devices with ARM Ubuntu, Bazel needs to
    be built from source.

    ```bash
    # For Bazel 3.0.0
    wget https://github.com/bazelbuild/bazel/releases/download/3.0.0/bazel-3.0.0-dist.zip
    sudo apt-get install build-essential openjdk-8-jdk python zip unzip
    unzip bazel-3.0.0-dist.zip
    env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
    sudo cp output/bazel /usr/local/bin/
    ```

3.  Install OpenCV and FFmpeg.

    Option 1. Use package manager tool to install the pre-compiled OpenCV
    libraries. FFmpeg will be installed via libopencv-video-dev.

    Note: Debian 9 and Ubuntu 16.04 provide OpenCV 2.4.9. You may want to take
    option 2 or 3 to install OpenCV 3 or above.

    ```bash
    $ sudo apt-get install libopencv-core-dev libopencv-highgui-dev \
                           libopencv-calib3d-dev libopencv-features2d-dev \
                           libopencv-imgproc-dev libopencv-video-dev
    ```

    Debian 9 and Ubuntu 18.04 install the packages in
    `/usr/lib/x86_64-linux-gnu`. MediaPipe's [`opencv_linux.BUILD`] and
    [`ffmpeg_linux.BUILD`] are configured for this library path. Ubuntu 20.04
    may install the OpenCV and FFmpeg packages in `/usr/local`, Please follow
    the option 3 below to modify the [`WORKSPACE`], [`opencv_linux.BUILD`] and
    [`ffmpeg_linux.BUILD`] files accordingly.

    Moreover, for Nvidia Jetson and Raspberry Pi devices with ARM Ubuntu, the
    library path needs to be modified like the following:

    ```bash
    sed -i "s/x86_64-linux-gnu/aarch64-linux-gnu/g" third_party/opencv_linux.BUILD
    ```

    Option 2. Run [`setup_opencv.sh`] to automatically build OpenCV from source
    and modify MediaPipe's OpenCV config.

    Option 3. Follow OpenCV's
    [documentation](https://docs.opencv.org/3.4.6/d7/d9f/tutorial_linux_install.html)
    to manually build OpenCV from source code.

    Note: You may need to modify [`WORKSPACE`], [`opencv_linux.BUILD`] and
    [`ffmpeg_linux.BUILD`] to point MediaPipe to your own OpenCV and FFmpeg
    libraries. For example if OpenCV and FFmpeg are both manually installed in
    "/usr/local/", you will need to update: (1) the "linux_opencv" and
    "linux_ffmpeg" new_local_repository rules in [`WORKSPACE`], (2) the "opencv"
    cc_library rule in [`opencv_linux.BUILD`], and (3) the "libffmpeg"
    cc_library rule in [`ffmpeg_linux.BUILD`]. These 3 changes are shown below:

    ```bash
    new_local_repository(
        name = "linux_opencv",
        build_file = "@//third_party:opencv_linux.BUILD",
        path = "/usr/local",
    )

    new_local_repository(
        name = "linux_ffmpeg",
        build_file = "@//third_party:ffmpeg_linux.BUILD",
        path = "/usr/local",
    )

    cc_library(
        name = "opencv",
        srcs = glob(
            [
                "lib/libopencv_core.so",
                "lib/libopencv_highgui.so",
                "lib/libopencv_imgcodecs.so",
                "lib/libopencv_imgproc.so",
                "lib/libopencv_video.so",
                "lib/libopencv_videoio.so",
            ],
        ),
        hdrs = glob([
            # For OpenCV 3.x
            "include/opencv2/**/*.h*",
            # For OpenCV 4.x
            # "include/opencv4/opencv2/**/*.h*",
        ]),
        includes = [
            # For OpenCV 3.x
            "include/",
            # For OpenCV 4.x
            # "include/opencv4/",
        ],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "libffmpeg",
        srcs = glob(
            [
                "lib/libav*.so",
            ],
        ),
        hdrs = glob(["include/libav*/*.h"]),
        includes = ["include"],
        linkopts = [
            "-lavcodec",
            "-lavformat",
            "-lavutil",
        ],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )
    ```

4.  For running desktop examples on Linux only (not on OS X) with GPU
    acceleration.

    ```bash
    # Requires a GPU with EGL driver support.
    # Can use mesa GPU libraries for desktop, (or Nvidia/AMD equivalent).
    sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev

    # To compile with GPU support, replace
    --define MEDIAPIPE_DISABLE_GPU=1
    # with
    --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11
    # when building GPU examples.
    ```

5.  Run the [Hello World desktop example](./hello_world_desktop.md).

    ```bash
    $ export GLOG_logtostderr=1

    # if you are running on Linux desktop with CPU only
    $ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
        mediapipe/examples/desktop/hello_world:hello_world

    # If you are running on Linux desktop with GPU support enabled (via mesa drivers)
    $ bazel run --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
        mediapipe/examples/desktop/hello_world:hello_world

    # Should print:
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    ```

If you run into a build error, please read
[Troubleshooting](./troubleshooting.md) to find the solutions of several common
build issues.

## Installing on CentOS

**Disclaimer**: Running MediaPipe on CentOS is experimental.

1.  Checkout MediaPipe repository.

    ```bash
    $ git clone https://github.com/google/mediapipe.git

    # Change directory into MediaPipe root directory
    $ cd mediapipe
    ```

2.  Install Bazel.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-redhat.html)
    to install Bazel 2.0 or higher.

3.  Install OpenCV.

    Option 1. Use package manager tool to install the pre-compiled version.

    Note: yum installs OpenCV 2.4.5, which may have an opencv/gstreamer
    [issue](https://github.com/opencv/opencv/issues/4592).

    ```bash
    $ sudo yum install opencv-devel
    ```

    Option 2. Build OpenCV from source code.

    Note: You may need to modify [`WORKSPACE`], [`opencv_linux.BUILD`] and
    [`ffmpeg_linux.BUILD`] to point MediaPipe to your own OpenCV and FFmpeg
    libraries. For example if OpenCV and FFmpeg are both manually installed in
    "/usr/local/", you will need to update: (1) the "linux_opencv" and
    "linux_ffmpeg" new_local_repository rules in [`WORKSPACE`], (2) the "opencv"
    cc_library rule in [`opencv_linux.BUILD`], and (3) the "libffmpeg"
    cc_library rule in [`ffmpeg_linux.BUILD`]. These 3 changes are shown below:

    ```bash
    new_local_repository(
        name = "linux_opencv",
        build_file = "@//third_party:opencv_linux.BUILD",
        path = "/usr/local",
    )

    new_local_repository(
        name = "linux_ffmpeg",
        build_file = "@//third_party:ffmpeg_linux.BUILD",
        path = "/usr/local",
    )

    cc_library(
        name = "opencv",
        srcs = glob(
            [
                "lib/libopencv_core.so",
                "lib/libopencv_highgui.so",
                "lib/libopencv_imgcodecs.so",
                "lib/libopencv_imgproc.so",
                "lib/libopencv_video.so",
                "lib/libopencv_videoio.so",
            ],
        ),
        hdrs = glob([
            # For OpenCV 3.x
            "include/opencv2/**/*.h*",
            # For OpenCV 4.x
            # "include/opencv4/opencv2/**/*.h*",
        ]),
        includes = [
            # For OpenCV 3.x
            "include/",
            # For OpenCV 4.x
            # "include/opencv4/",
        ],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "libffmpeg",
        srcs = glob(
            [
                "lib/libav*.so",
            ],
        ),
        hdrs = glob(["include/libav*/*.h"]),
        includes = ["include"],
        linkopts = [
            "-lavcodec",
            "-lavformat",
            "-lavutil",
        ],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )
    ```

4.  Run the [Hello World desktop example](./hello_world_desktop.md).

    ```bash
    $ export GLOG_logtostderr=1
    # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' if you are running on Linux desktop with CPU only
    $ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
        mediapipe/examples/desktop/hello_world:hello_world

    # Should print:
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    ```

If you run into a build error, please read
[Troubleshooting](./troubleshooting.md) to find the solutions of several common
build issues.

## Installing on macOS

1.  Prework:

    *   Install [Homebrew](https://brew.sh).
    *   Install [Xcode](https://developer.apple.com/xcode/) and its Command Line
        Tools by `xcode-select --install`.

2.  Checkout MediaPipe repository.

    ```bash
    $ git clone https://github.com/google/mediapipe.git

    $ cd mediapipe
    ```

3.  Install Bazel.

    Option 1. Use package manager tool to install Bazel

    ```bash
    $ brew install bazel
    # Run 'bazel version' to check version of bazel
    ```

    Option 2. Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x)
    to install Bazel 2.0 or higher.

4.  Install OpenCV and FFmpeg.

    Option 1. Use HomeBrew package manager tool to install the pre-compiled
    OpenCV 3.4.5 libraries. FFmpeg will be installed via OpenCV.

    ```bash
    $ brew install opencv@3

    # There is a known issue caused by the glog dependency. Uninstall glog.
    $ brew uninstall --ignore-dependencies glog
    ```

    Option 2. Use MacPorts package manager tool to install the OpenCV libraries.

    ```bash
    $ port install opencv
    ```

    Note: when using MacPorts, please edit the [`WORKSPACE`],
    [`opencv_macos.BUILD`], and [`ffmpeg_macos.BUILD`] files like the following:

    ```bash
    new_local_repository(
        name = "macos_opencv",
        build_file = "@//third_party:opencv_macos.BUILD",
        path = "/opt",
    )

    new_local_repository(
        name = "macos_ffmpeg",
        build_file = "@//third_party:ffmpeg_macos.BUILD",
        path = "/opt",
    )

    cc_library(
        name = "opencv",
        srcs = glob(
            [
                "local/lib/libopencv_core.dylib",
                "local/lib/libopencv_highgui.dylib",
                "local/lib/libopencv_imgcodecs.dylib",
                "local/lib/libopencv_imgproc.dylib",
                "local/lib/libopencv_video.dylib",
                "local/lib/libopencv_videoio.dylib",
            ],
        ),
        hdrs = glob(["local/include/opencv2/**/*.h*"]),
        includes = ["local/include/"],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "libffmpeg",
        srcs = glob(
            [
                "local/lib/libav*.dylib",
            ],
        ),
        hdrs = glob(["local/include/libav*/*.h"]),
        includes = ["local/include/"],
        linkopts = [
            "-lavcodec",
            "-lavformat",
            "-lavutil",
        ],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )

    ```

5.  Make sure that Python 3 and the Python "six" library are installed.

    ```
    $ brew install python
    $ sudo ln -s -f /usr/local/bin/python3.7 /usr/local/bin/python
    $ python --version
    Python 3.7.4
    $ pip3 install --user six
    ```

6.  Run the [Hello World desktop example](./hello_world_desktop.md).

    ```bash
    $ export GLOG_logtostderr=1
    # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
    $ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
        mediapipe/examples/desktop/hello_world:hello_world

    # Should print:
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    ```

If you run into a build error, please read
[Troubleshooting](./troubleshooting.md) to find the solutions of several common
build issues.

## Installing on Windows

**Disclaimer**: Running MediaPipe on Windows is experimental.

Note: building MediaPipe Android apps is still not possible on native
Windows. Please do this in WSL instead and see the WSL setup instruction in the
next section.

1.  Install [MSYS2](https://www.msys2.org/) and edit the `%PATH%` environment
    variable.

    If MSYS2 is installed to `C:\msys64`, add `C:\msys64\usr\bin` to your
    `%PATH%` environment variable.

2.  Install necessary packages.

    ```
    C:\> pacman -S git patch unzip
    ```

3.  Install Python and allow the executable to edit the `%PATH%` environment
    variable.

    Download Python Windows executable from
    https://www.python.org/downloads/windows/ and install.

4.  Install Visual C++ Build Tools 2019 and WinSDK

    Go to https://visualstudio.microsoft.com/visual-cpp-build-tools, download
    build tools, and install Microsoft Visual C++ 2019 Redistributable and
    Microsoft Build Tools 2019.

    Download the WinSDK from
    https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/ and
    install.

5.  Install Bazel and add the location of the Bazel executable to the `%PATH%`
    environment variable.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-windows.html)
    to install Bazel 2.0 or higher.

6.  Set Bazel variables.

    ```
    # Find the exact paths and version numbers from your local version.
    C:\> set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
    C:\> set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
    C:\> set BAZEL_VC_FULL_VERSION=14.25.28610
    C:\> set BAZEL_WINSDK_FULL_VERSION=10.1.18362.1
    ```

7.  Checkout MediaPipe repository.

    ```
    C:\Users\Username\mediapipe_repo> git clone https://github.com/google/mediapipe.git

    # Change directory into MediaPipe root directory
    C:\Users\Username\mediapipe_repo> cd mediapipe
    ```

8.  Install OpenCV.

    Download the Windows executable from https://opencv.org/releases/ and
    install. We currently use OpenCV 3.4.10. Remember to edit the [`WORKSPACE`]
    file if OpenCV is not installed at `C:\opencv`.

    ```
    new_local_repository(
        name = "windows_opencv",
        build_file = "@//third_party:opencv_windows.BUILD",
        path = "C:\\<path to opencv>\\build",
    )
    ```

9.  Run the [Hello World desktop example](./hello_world_desktop.md).

    Note: For building MediaPipe on Windows, please add `--action_env
    PYTHON_BIN_PATH="C://path//to//python.exe"` to the build command.
    Alternatively, you can follow
    [issue 724](https://github.com/google/mediapipe/issues/724) to fix the
    python configuration manually.

    ```
    C:\Users\Username\mediapipe_repo>bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://python_36//python.exe" mediapipe/examples/desktop/hello_world

    C:\Users\Username\mediapipe_repo>set GLOG_logtostderr=1

    C:\Users\Username\mediapipe_repo>bazel-bin\mediapipe\examples\desktop\hello_world\hello_world.exe

    # should print:
    # I20200514 20:43:12.277598  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.278597  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
    # I20200514 20:43:12.280613  1200 hello_world.cc:56] Hello World!

    ```

If you run into a build error, please read
[Troubleshooting](./troubleshooting.md) to find the solutions of several common
build issues.

## Installing on Windows Subsystem for Linux (WSL)

Note: The pre-built OpenCV packages don't support cameras in WSL. Unless you
[compile](https://funvision.blogspot.com/2019/12/opencv-web-camera-and-video-streams-in.html)
OpenCV with FFMPEG and GStreamer in WSL, the live demos won't work with any
cameras. Alternatively, you use a video file as input.

1.  Follow the
    [instruction](https://docs.microsoft.com/en-us/windows/wsl/install-win10) to
    install Windows Sysystem for Linux (Ubuntu).

2.  Install Windows ADB and start the ADB server in Windows.

    Note: Windows' and WSL’s adb versions must be the same version, e.g., if WSL
    has ADB 1.0.39, you need to download the corresponding Windows ADB from
    [here](https://dl.google.com/android/repository/platform-tools_r26.0.1-windows.zip).

3.  Launch WSL.

    Note: All the following steps will be executed in WSL. The Windows directory
    of the Linux Subsystem can be found in
    C:\Users\YourUsername\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_SomeID\LocalState\rootfs\home

4.  Install the needed packages.

    ```bash
    username@DESKTOP-TMVLBJ1:~$ sudo apt-get update && sudo apt-get install -y build-essential git python zip adb openjdk-8-jdk
    ```

5.  Install Bazel.

    ```bash
    username@DESKTOP-TMVLBJ1:~$ curl -sLO --retry 5 --retry-max-time 10 \
    https://storage.googleapis.com/bazel/3.0.0/release/bazel-3.0.0-installer-linux-x86_64.sh && \
    sudo mkdir -p /usr/local/bazel/3.0.0 && \
    chmod 755 bazel-3.0.0-installer-linux-x86_64.sh && \
    sudo ./bazel-3.0.0-installer-linux-x86_64.sh --prefix=/usr/local/bazel/3.0.0 && \
    source /usr/local/bazel/3.0.0/lib/bazel/bin/bazel-complete.bash

    username@DESKTOP-TMVLBJ1:~$ /usr/local/bazel/3.0.0/lib/bazel/bin/bazel version && \
    alias bazel='/usr/local/bazel/3.0.0/lib/bazel/bin/bazel'
    ```

6.  Checkout MediaPipe repository.

    ```bash
    username@DESKTOP-TMVLBJ1:~$ git clone https://github.com/google/mediapipe.git

    username@DESKTOP-TMVLBJ1:~$ cd mediapipe
    ```

7.  Install OpenCV and FFmpeg.

    Option 1. Use package manager tool to install the pre-compiled OpenCV
    libraries. FFmpeg will be installed via libopencv-video-dev.

    ```bash
    username@DESKTOP-TMVLBJ1:~/mediapipe$ sudo apt-get install libopencv-core-dev libopencv-highgui-dev \
                           libopencv-calib3d-dev libopencv-features2d-dev \
                           libopencv-imgproc-dev libopencv-video-dev
    ```

    Option 2. Run [`setup_opencv.sh`] to automatically build OpenCV from source
    and modify MediaPipe's OpenCV config.

    Option 3. Follow OpenCV's
    [documentation](https://docs.opencv.org/3.4.6/d7/d9f/tutorial_linux_install.html)
    to manually build OpenCV from source code.

    Note: You may need to modify [`WORKSPACE`] and [`opencv_linux.BUILD`] to
    point MediaPipe to your own OpenCV libraries, e.g., if OpenCV 4 is installed
    in "/usr/local/", you need to update the "linux_opencv" new_local_repository
    rule in [`WORKSPACE`] and "opencv" cc_library rule in [`opencv_linux.BUILD`]
    like the following:

    ```bash
    new_local_repository(
        name = "linux_opencv",
        build_file = "@//third_party:opencv_linux.BUILD",
        path = "/usr/local",
    )

    cc_library(
        name = "opencv",
        srcs = glob(
            [
                "lib/libopencv_core.so",
                "lib/libopencv_highgui.so",
                "lib/libopencv_imgcodecs.so",
                "lib/libopencv_imgproc.so",
                "lib/libopencv_video.so",
                "lib/libopencv_videoio.so",
            ],
        ),
        hdrs = glob(["include/opencv4/**/*.h*"]),
        includes = ["include/opencv4/"],
        linkstatic = 1,
        visibility = ["//visibility:public"],
    )
    ```

8.  Run the [Hello World desktop example](./hello_world_desktop.md).

    ```bash
    username@DESKTOP-TMVLBJ1:~/mediapipe$ export GLOG_logtostderr=1

    # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
    username@DESKTOP-TMVLBJ1:~/mediapipe$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
        mediapipe/examples/desktop/hello_world:hello_world

    # Should print:
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    ```

If you run into a build error, please
read [Troubleshooting](./troubleshooting.md) to find the solutions of several
common build issues.

## Installing using Docker

This will use a Docker image that will isolate mediapipe's installation from the rest of the system.

1.  [Install Docker](https://docs.docker.com/install/#supported-platforms) on
    your host system.

2.  Build a docker image with tag "mediapipe".

    ```bash
    $ git clone https://github.com/google/mediapipe.git
    $ cd mediapipe
    $ docker build --tag=mediapipe .

    # Should print:
    # Sending build context to Docker daemon  147.8MB
    # Step 1/9 : FROM ubuntu:latest
    # latest: Pulling from library/ubuntu
    # 6abc03819f3e: Pull complete
    # 05731e63f211: Pull complete
    # ........
    # See http://bazel.build/docs/getting-started.html to start a new project!
    # Removing intermediate container 82901b5e79fa
    # ---> f5d5f402071b
    # Step 9/9 : COPY . /mediapipe/
    # ---> a95c212089c5
    # Successfully built a95c212089c5
    # Successfully tagged mediapipe:latest
    ```

3.  Run the [Hello World desktop example](./hello_world_desktop.md).

    ```bash
    $ docker run -it --name mediapipe mediapipe:latest

    root@bca08b91ff63:/mediapipe# GLOG_logtostderr=1 bazel run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world

    # Should print:
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    # Hello World!
    ```

If you run into a build error, please
read [Troubleshooting](./troubleshooting.md) to find the solutions of several
common build issues.

4.  Build a MediaPipe Android example.

    ```bash
    $ docker run -it --name mediapipe mediapipe:latest

    root@bca08b91ff63:/mediapipe# bash ./setup_android_sdk_and_ndk.sh

    # Should print:
    # Android NDK is now installed. Consider setting $ANDROID_NDK_HOME environment variable to be /root/Android/Sdk/ndk-bundle/android-ndk-r18b
    # Set android_ndk_repository and android_sdk_repository in WORKSPACE
    # Done

    root@bca08b91ff63:/mediapipe# bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu:objectdetectiongpu

    # Should print:
    # Target //mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu:objectdetectiongpu up-to-date:
    # bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu/objectdetectiongpu_deploy.jar
    # bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu/objectdetectiongpu_unsigned.apk
    # bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu/objectdetectiongpu.apk
    # INFO: Elapsed time: 144.462s, Critical Path: 79.47s
    # INFO: 1958 processes: 1 local, 1863 processwrapper-sandbox, 94 worker.
    # INFO: Build completed successfully, 2028 total actions
    ```

<!-- 5.  Uncomment the last line of the Dockerfile

    ```bash
    RUN bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/demo:object_detection_tensorflow_demo
    ```

    and rebuild the image and then run the docker image

    ```bash
    docker build --tag=mediapipe .
    docker run -i -t mediapipe:latest
    ``` -->

[`WORKSPACE`]: https://github.com/google/mediapipe/blob/master/WORKSPACE
[`opencv_linux.BUILD`]: https://github.com/google/mediapipe/tree/master/third_party/opencv_linux.BUILD
[`ffmpeg_linux.BUILD`]:https://github.com/google/mediapipe/tree/master/third_party/ffmpeg_linux.BUILD
[`opencv_macos.BUILD`]: https://github.com/google/mediapipe/tree/master/third_party/opencv_macos.BUILD
[`ffmpeg_macos.BUILD`]:https://github.com/google/mediapipe/tree/master/third_party/ffmpeg_macos.BUILD
[`setup_opencv.sh`]: https://github.com/google/mediapipe/blob/master/setup_opencv.sh
