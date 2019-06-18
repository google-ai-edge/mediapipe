## Installing MediaPipe

Choose your operating system:

-   [Dependences](#dependences)
-   [Installing on Debian and Ubuntu](#installing-on-debian-and-ubuntu)
-   [Installing on CentOS](#installing-on-centos)
-   [Installing on macOS](#installing-on-macos)
-   [Installing using Docker](#installing-using-docker)
-   [Setting up Android SDK and NDK](#setting-up-android-sdk-and-ndk)

### Dependences

Required libraries

*   Prefer OpenCV 3.x and above but can work with OpenCV 2.x (deprecation in the
    future)

*   Bazel 0.23 and above

*   gcc and g++ version other than 6.3 and 7.3 (if you need TensorFlow
    calculators/demos)

*   Android SDK release 28.0.3 and above

*   Android NDK r18b and above

### Installing on Debian and Ubuntu

1.  Checkout mediapipe repository

    ```bash
    $ git clone https://github.com/google/mediapipe/mediapipe.git

    # Change directory into mediapipe root directory
    $ cd mediapipe
    ```

2.  Install Bazel

    Option 1. Use package manager tool to install the latest version of Bazel.

    ```bash
    $ sudo apt-get install bazel

    # Run 'bazel version' to check version of bazel installed
    ```

    Option 2. Follow Bazel's
    [documentation](https://docs.bazel.build/versions/master/install-ubuntu.html)
    to install any version of Bazel manually.

3.  Install OpenCV

    Option 1. Use package manager tool to install the pre-compiled OpenCV
    libraries.

    Note: Debian 9 and Ubuntu 16.04 provide OpenCV 2.4.9. You may want to
    take option 2 or 3 to install OpenCV 3 or above.

    ```bash
    $ sudo apt-get install libopencv-core-dev libopencv-highgui-dev \
                           libopencv-imgproc-dev libopencv-video-dev
    ```

    Option 2. Run [`setup_opencv.sh`] to automatically build OpenCV from source
    and modify MediaPipe's OpenCV config.

    Option 3. Follow OpenCV's
    [documentation](https://docs.opencv.org/3.4.6/d7/d9f/tutorial_linux_install.html)
    to manually build OpenCV from source code.

    Note: You may need to modify [`WORKSAPCE`] and [`opencv_linux.BUILD`] to point
    MediaPipe to your own OpenCV libraries, e.g., if OpenCV 4 is
    installed in "/usr/local/", you need to update the "linux_opencv"
    new_local_repository rule in [`WORKSAPCE`] and "opencv" cc_library rule in
    [`opencv_linux.BUILD`] like the following:

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
              "lib/libopencv_core.so*",
              "lib/libopencv_highgui.so*",
              "lib/libopencv_imgcodecs.so*",
              "lib/libopencv_imgproc.so*",
              "lib/libopencv_video.so*",
              "lib/libopencv_videoio.so*",

          ],
      ),
      hdrs = glob(["include/opencv4/**/*.h*"]),
      includes = ["include/opencv4/"],
      linkstatic = 1,
      visibility = ["//visibility:public"],
    )

    ```

4.  Run the hello world desktop example

    ```bash
    # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
    $ bazel run --define 'MEDIAPIPE_DISABLE_GPU=1' \
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

### Installing on CentOS

1.  Checkout mediapipe repository

    ```bash
    $ git clone https://github.com/google/mediapipe/mediapipe.git

    # Change directory into mediapipe root directory
    $ cd mediapipe
    ```

2.  Install Bazel

    Follow Bazel's
    [documentation](https://docs.bazel.build/versions/master/install-redhat.html)
    to install Bazel manually.

3.  Install OpenCV

    Option 1. Use package manager tool to install the pre-compiled version.

    Note: yum installs OpenCV 2.4.5, which may have an opencv/gstreamer
    [issue](https://github.com/opencv/opencv/issues/4592).

    ```bash
    $ sudo yum install opencv-devel
    ```

    Option 2. Build OpenCV from source code.

    Note: You may need to modify [`WORKSAPCE`] and [`opencv_linux.BUILD`] to point
    MediaPipe to your own OpenCV libraries, e.g., if OpenCV 4 is
    installed in "/usr/local/", you need to update the "linux_opencv"
    new_local_repository rule in [`WORKSAPCE`] and "opencv" cc_library rule in
    [`opencv_linux.BUILD`] like the following:

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
              "lib/libopencv_core.so*",
              "lib/libopencv_highgui.so*",
              "lib/libopencv_imgcodecs.so*",
              "lib/libopencv_imgproc.so*",
              "lib/libopencv_video.so*",
              "lib/libopencv_videoio.so*",

          ],
      ),
      hdrs = glob(["include/opencv4/**/*.h*"]),
      includes = ["include/opencv4/"],
      linkstatic = 1,
      visibility = ["//visibility:public"],
    )

    ```

4.  Run the hello world desktop example

    ```bash
    # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
    $ bazel run --define 'MEDIAPIPE_DISABLE_GPU=1' \
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

### Installing on macOS

1.  Checkout mediapipe repository

    ```bash
    $ git clone https://github.com/google/mediapipe/mediapipe.git

    $ cd mediapipe
    ```

2.  Install Bazel

    Option 1. Use package manager tool to install the latest version of Bazel.

    ```bash
    $ brew install bazel

    # Run 'bazel version' to check version of bazel installed
    ```

    Option 2. Follow Bazel's
    [documentation](https://docs.bazel.build/versions/master/install-ubuntu.html)
    to install any version of Bazel manually.

3.  Install OpenCV

    Option 1. Use HomeBrew package manager tool to install the pre-compiled
    OpenCV libraries.

    ```bash
    $ brew install opencv
    ```

    Option 2. Use MacPorts package manager tool to install the OpenCV libraries.

    ```bash
    $ port install opencv
    ```

    Note: when using MacPorts, please edit the [`WORKSAPCE`] and
    [`opencv_linux.BUILD`] files like the following:

    ```bash
    new_local_repository(
      name = "macos_opencv",
      build_file = "@//third_party:opencv_macos.BUILD",
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
    ```

4.  Run the hello world desktop example

    ```bash
    # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
    $ bazel run --define 'MEDIAPIPE_DISABLE_GPU=1' \
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

### Installing using Docker

This will use a Docker image that will isolate mediapipe's installation from the rest of the system.

1.  [Install Docker](https://docs.docker.com/install/#supported-platforms) on
    your host sytem

2.  Build a docker image with tag "mediapipe"

    ```bash
    $ git clone https://github.com/google/mediapipe/mediapipe.git
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

3.  Run the hello world desktop example in docker

    ```bash
    $ docker run -it --name mediapipe mediapipe:latest

    root@bca08b91ff63:/mediapipe# bazel run --define 'MEDIAPIPE_DISABLE_GPU=1' mediapipe/examples/desktop/hello_world:hello_world

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

<!-- 4.  Uncomment the last line of the Dockerfile

    ```bash
    RUN bazel build -c opt --define 'MEDIAPIPE_DISABLE_GPU=1' mediapipe/examples/desktop/demo:object_detection_tensorflow_demo
    ```

    and rebuild the image and then run the docker image

    ```bash
    docker build --tag=mediapipe .
    docker run -i -t mediapipe:latest
    ``` -->


### Setting up Android Studio with MediaPipe

The steps below use Android Studio to build and install a MediaPipe demo app.

1.  Install and launch android studio.

2.  Select `Configure` | `SDK Manager` | `SDK Platforms`

    *   verify that an Android SDK is installed
    *   note the Android SDK Location such as `/usr/local/home/Android/Sdk`

3.  Select `Configure` | `SDK Manager` | `SDK Tools`

    *   verify that an Android NDK is installed
    *   note the Android NDK Location such as `/usr/local/home/Android/Sdk/ndk-bundle`

4.  Set environment variables `$ANDROID_HOME` and `$ANDROID_NDK_HOME` to point to
    the installed SDK and NDK.

    ```bash
    export ANDROID_HOME=/usr/local/home/Android/Sdk
    export ANDROID_NDK_HOME=/usr/local/home/Android/Sdk/ndk-bundle
    ```

5.  Select `Configure` | `Plugins` install `Bazel`.

6.  Select `Import Bazel Project`

    *   select `Workspace`: `/path/to/mediapipe`
    *   select `Generate from BUILD file`: `/path/to/mediapipe/BUILD`
    *   select `Finish`

7.  Connect an android device to the workstation.

8.  Select `Run...` | `Edit Configurations...`

    *   enter Target Expression:
        `//mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu`
    *   enter Bazel command: `mobile-install`
    *   enter Bazel flags: `-c opt --config=android_arm64` select `Run`

### Setting up Android SDK and NDK

If Android SDK and NDK are installed (likely by Android Studio), please set
$ANDROID_HOME and $ANDROID_NDK_HOME to point to the installed SDK and NDK.

```bash
export ANDROID_HOME=<path to the Android SDK>
export ANDROID_NDK_HOME=<path to the Android NDK>
```

Otherwise, please run [`setup_android_sdk_and_ndk.sh`] to download and setup
Android SDK and NDK for MediaPipe before building any Android demos.

[`WORKSAPCE`]: https://github.com/google/mediapipe/tree/master/WORKSPACE
[`opencv_linux.BUILD`]: https://github.com/google/mediapipe/tree/master/third_party/opencv_linux.BUILD
[`setup_opencv.sh`]: https://github.com/google/mediapipe/tree/master/setup_opencv.sh
[`setup_android_sdk_and_ndk.sh`]: https://github.com/google/mediapipe/tree/master/setup_android_sdk_and_ndk.sh
