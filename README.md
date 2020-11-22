# Kazakh-Russian Sign Language Recognition using Mediapipe and Tensorflow

## Installing on Debian and Ubuntu

1.  Checkout MediaPipe repository.

    ```bash
    $ git clone https://github.com/kurshakuz/mediapipe.git

    # Change directory into MediaPipe root directory
    $ cd mediapipe
    ```

2.  Install Bazel.

    Follow the official
    [Bazel documentation](https://docs.bazel.build/versions/master/install-ubuntu.html)
    to install Bazel 3.4 or higher.

    For Nvidia Jetson and Raspberry Pi devices with ARM Ubuntu, Bazel needs to
    be built from source.

    ```bash
    # For Bazel 3.4.0
    wget https://github.com/bazelbuild/bazel/releases/download/3.4.0/bazel-3.4.0-dist.zip
    sudo apt-get install build-essential openjdk-8-jdk python zip unzip
    unzip bazel-3.4.0-dist.zip
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

    Option 2. Run [`setup_opencv.sh`] to automatically build OpenCV from source
    and modify MediaPipe's OpenCV config.

    Option 3. Follow OpenCV's
    [documentation](https://docs.opencv.org/3.4.6/d7/d9f/tutorial_linux_install.html)
    to manually build OpenCV from source code.

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

## Building required hand tracking module on desktop

### Option 1: Running on CPU

1.  To build, for example, [MediaPipe Hands](../solutions/hands.md), run:

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
    ```

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible.

### Option 2: Running on GPU

Note: This currently works only on Linux, and please first follow
[OpenGL ES Setup on Linux Desktop](./gpu_support.md#opengl-es-setup-on-linux-desktop).

1.  To build, for example, [MediaPipe Hands](../solutions/hands.md), run:

    ```bash
    bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
      mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
    ```

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible, or GPU drivers not setup
    properly.

## Running the pretrained model for sign language recognition

Insert your data to any folder and pass it as a ```--output_data_path``` variable. The result will appear in the same folder and will be called `result.txt`.

```bash
cd RNN/

python3 predict.py --input_data_path='./test_video/' --output_data_path='./test_video_output/'
```
