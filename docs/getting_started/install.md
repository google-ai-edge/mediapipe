---
layout: default
title: Installation
parent: Getting Started
nav_order: 6
---

# Installation
{: .no_toc }

1. TOC
{:toc}
---

Note: To interoperate with OpenCV, OpenCV 3.x to 4.1 are preferred. OpenCV
2.x currently works but interoperability support may be deprecated in the
future.

Note: If you plan to use TensorFlow calculators and example apps, there is a
known issue with gcc and g++ version 6.3 and 7.3. Please use other versions.

Note: To make Mediapipe work with TensorFlow, please set Python 3.7 as the
default Python version and install the Python "six" library by running `pip3
install --user six`.

## Installing on Debian and Ubuntu

1.  Install Bazelisk.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-bazelisk.html)
    to install Bazelisk.

2.  Checkout MediaPipe repository.

    ```bash
    $ cd $HOME
    $ git clone https://github.com/google/mediapipe.git

    # Change directory into MediaPipe root directory
    $ cd mediapipe
    ```

3.  Install OpenCV and FFmpeg.

    **Option 1**. Use package manager tool to install the pre-compiled OpenCV
    libraries. FFmpeg will be installed via `libopencv-video-dev`.

    OS                   | OpenCV
    -------------------- | ------
    Debian 9 (stretch)   | 2.4
    Debian 10 (buster)   | 3.2
    Debian 11 (bullseye) | 4.5
    Ubuntu 16.04 LTS     | 2.4
    Ubuntu 18.04 LTS     | 3.2
    Ubuntu 20.04 LTS     | 4.2
    Ubuntu 20.04 LTS     | 4.2
    Ubuntu 21.04         | 4.5

    ```bash
    $ sudo apt-get install -y \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-calib3d-dev \
        libopencv-features2d-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev
    ```

    MediaPipe's [`opencv_linux.BUILD`] and [`WORKSPACE`] are already configured
    for OpenCV 2/3 and should work correctly on any architecture:

    ```bash
    # WORKSPACE
    new_local_repository(
      name = "linux_opencv",
      build_file = "@//third_party:opencv_linux.BUILD",
      path = "/usr",
    )

    # opencv_linux.BUILD for OpenCV 2/3 installed from Debian package
    cc_library(
      name = "opencv",
      linkopts = [
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
      ],
    )
    ```

    For OpenCV 4 you need to modify [`opencv_linux.BUILD`] taking into account
    current architecture:

    ```bash
    # WORKSPACE
    new_local_repository(
      name = "linux_opencv",
      build_file = "@//third_party:opencv_linux.BUILD",
      path = "/usr",
    )

    # opencv_linux.BUILD for OpenCV 4 installed from Debian package
    cc_library(
      name = "opencv",
      hdrs = glob([
        # Uncomment according to your multiarch value (gcc -print-multiarch):
        #  "include/aarch64-linux-gnu/opencv4/opencv2/cvconfig.h",
        #  "include/arm-linux-gnueabihf/opencv4/opencv2/cvconfig.h",
        #  "include/x86_64-linux-gnu/opencv4/opencv2/cvconfig.h",
        "include/opencv4/opencv2/**/*.h*",
      ]),
      includes = [
        # Uncomment according to your multiarch value (gcc -print-multiarch):
        #  "include/aarch64-linux-gnu/opencv4/",
        #  "include/arm-linux-gnueabihf/opencv4/",
        #  "include/x86_64-linux-gnu/opencv4/",
        "include/opencv4/",
      ],
      linkopts = [
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
      ],
    )
    ```

    **Option 2**. Run [`setup_opencv.sh`] to automatically build OpenCV from
    source and modify MediaPipe's OpenCV config. This option will do all steps
    defined in Option 3 automatically.

    **Option 3**. Follow OpenCV's
    [documentation](https://docs.opencv.org/3.4.6/d7/d9f/tutorial_linux_install.html)
    to manually build OpenCV from source code.

    You may need to modify [`WORKSPACE`] and [`opencv_linux.BUILD`] to point
    MediaPipe to your own OpenCV libraries. Assume OpenCV would be installed to
    `/usr/local/` which is recommended by default.

    OpenCV 2/3 setup:

    ```bash
    # WORKSPACE
    new_local_repository(
      name = "linux_opencv",
      build_file = "@//third_party:opencv_linux.BUILD",
      path = "/usr/local",
    )

    # opencv_linux.BUILD for OpenCV 2/3 installed to /usr/local
    cc_library(
      name = "opencv",
      linkopts = [
        "-L/usr/local/lib",
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
      ],
    )
    ```

    OpenCV 4 setup:

    ```bash
    # WORKSPACE
    new_local_repository(
      name = "linux_opencv",
      build_file = "@//third_party:opencv_linux.BUILD",
      path = "/usr/local",
    )

    # opencv_linux.BUILD for OpenCV 4 installed to /usr/local
    cc_library(
      name = "opencv",
      hdrs = glob([
        "include/opencv4/opencv2/**/*.h*",
      ]),
      includes = [
        "include/opencv4/",
      ],
      linkopts = [
        "-L/usr/local/lib",
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
      ],
    )
    ```

    Current FFmpeg setup is defined in [`ffmpeg_linux.BUILD`] and should work
    for any architecture:

    ```bash
    # WORKSPACE
    new_local_repository(
      name = "linux_ffmpeg",
      build_file = "@//third_party:ffmpeg_linux.BUILD",
      path = "/usr"
    )

    # ffmpeg_linux.BUILD for FFmpeg installed from Debian package
    cc_library(
      name = "libffmpeg",
      linkopts = [
        "-l:libavcodec.so",
        "-l:libavformat.so",
        "-l:libavutil.so",
      ],
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

5.  Run the [Hello World! in C++ example](./hello_world_cpp.md).

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

1.  Install Bazelisk.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-bazelisk.html)
    to install Bazelisk.

2.  Checkout MediaPipe repository.

    ```bash
    $ git clone https://github.com/google/mediapipe.git

    # Change directory into MediaPipe root directory
    $ cd mediapipe
    ```

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

4.  Run the [Hello World! in C++ example](./hello_world_cpp.md).

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

2.  Install Bazelisk.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-bazelisk.html)
    to install Bazelisk.

3.  Checkout MediaPipe repository.

    ```bash
    $ git clone https://github.com/google/mediapipe.git

    $ cd mediapipe
    ```

4.  Install OpenCV and FFmpeg.

    Option 1. Use HomeBrew package manager tool to install the pre-compiled
    OpenCV 3 libraries. FFmpeg will be installed via OpenCV.

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

6.  Run the [Hello World! in C++ example](./hello_world_cpp.md).

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

    Go to
    [the VisualStudio website](https://visualstudio.microsoft.com/visual-cpp-build-tools),
    download build tools, and install Microsoft Visual C++ 2019 Redistributable
    and Microsoft Build Tools 2019.

    Download the WinSDK from
    [the official MicroSoft website](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
    and install.

5.  Install Bazel or Bazelisk and add the location of the Bazel executable to
    the `%PATH%` environment variable.

    Option 1. Follow
    [the official Bazel documentation](https://docs.bazel.build/versions/master/install-windows.html)
    to install Bazel 5.2.0 or higher.

    Option 2. Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-bazelisk.html)
    to install Bazelisk.

6.  Set Bazel variables. Learn more details about
    ["Build on Windows"](https://docs.bazel.build/versions/master/windows.html#build-c-with-msvc)
    in the Bazel official documentation.

    ```
    # Please find the exact paths and version numbers from your local version.
    C:\> set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
    C:\> set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
    C:\> set BAZEL_VC_FULL_VERSION=<Your local VC version>
    C:\> set BAZEL_WINSDK_FULL_VERSION=<Your local WinSDK version>
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

9.  Run the [Hello World! in C++ example](./hello_world_cpp.md).

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
    [here](https://dl.google.com/android/repository/platform-tools_r30.0.3-windows.zip).

3.  Launch WSL.

    Note: All the following steps will be executed in WSL. The Windows directory
    of the Linux Subsystem can be found in
    C:\Users\YourUsername\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_SomeID\LocalState\rootfs\home

4.  Install the needed packages.

    ```bash
    username@DESKTOP-TMVLBJ1:~$ sudo apt-get update && sudo apt-get install -y build-essential git python zip adb openjdk-8-jdk
    ```

5.  Install Bazelisk.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-bazelisk.html)
    to install Bazelisk.

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

8.  Run the [Hello World! in C++ example](./hello_world_cpp.md).

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

3.  Run the [Hello World! in C++ example](./hello_world_cpp.md).

    ```bash
    $ docker run -it --name mediapipe mediapipe:latest

    root@bca08b91ff63:/mediapipe# GLOG_logtostderr=1 bazel run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world

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
    # Android NDK is now installed. Consider setting $ANDROID_NDK_HOME environment variable to be /root/Android/Sdk/ndk-bundle/android-ndk-r19c
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
