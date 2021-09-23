# Coral Support

## Bazel Setup

You can compile MediaPipe with enabled Edge TPU support to run
[Coral models](http://coral.ai/models). Just add
`--define MEDIAPIPE_EDGE_TPU=<type>` to the `bazel` command:

* `--define MEDIAPIPE_EDGE_TPU=usb` for Coral USB devices on Linux and macOS
* `--define MEDIAPIPE_EDGE_TPU=pci` for Coral PCIe devices on Linux
* `--define MEDIAPIPE_EDGE_TPU=all` for both Coral USB and PCIe devices on Linux

You have to install `libusb` library in order to compile with USB support:

* `libusb-1.0-0-dev` on Linux
* `libusb` on macOS via MacPorts or Homebrew

Command to compile face detection Coral example:

```bash
bazel build \
  --compilation_mode=opt \
  --define darwinn_portable=1 \
  --define MEDIAPIPE_DISABLE_GPU=1 \
  --define MEDIAPIPE_EDGE_TPU=usb \
  --linkopt=-l:libusb-1.0.so \
  mediapipe/examples/coral:face_detection_tpu build
```

## Cross-compilation

Sometimes you need to cross-compile MediaPipe source code, e.g. get `ARM32`
or `ARM64` binaries on `x86` system. Install cross-compilation toolchain on
your system or use our preconfigured Docker environment for that:

```bash
# For ARM32 (e.g. Raspberry Pi)
make -C mediapipe/examples/coral PLATFORM=armhf docker

# For ARM64 (e.g. Coral Dev Board)
make -C mediapipe/examples/coral PLATFORM=arm64 docker
```

After running this command you'll get a shell to the Docker environment which
has everything ready to start compilation:

```bash
# For ARM32 (e.g. Raspberry Pi)
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=armv7a \
    --define darwinn_portable=1 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define MEDIAPIPE_EDGE_TPU=usb \
    --linkopt=-l:libusb-1.0.so \
    mediapipe/examples/coral:face_detection_tpu build

# For ARM64 (e.g. Coral Dev Board)
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=aarch64 \
    --define darwinn_portable=1 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define MEDIAPIPE_EDGE_TPU=usb \
    --linkopt=-l:libusb-1.0.so \
    mediapipe/examples/coral:face_detection_tpu build
```

Our Docker environment defines `${BAZEL_CPU}` value, so you can use it directly:

```bash
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=${BAZEL_CPU} \
    --define darwinn_portable=1 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define MEDIAPIPE_EDGE_TPU=usb \
    --linkopt=-l:libusb-1.0.so \
    mediapipe/examples/coral:face_detection_tpu build
```

The command above is already defined in our `Makefile`, so you can simply run:

```bash
make -C mediapipe/examples/coral \
     BAZEL_TARGET=mediapipe/examples/coral:face_detection_tpu \
     build
```

The output binary will be automatically copied to `out/<platform>` directory.

You can also run compilation inside Docker environment as a single
command:

```bash
make -C mediapipe/examples/coral \
     PLATFORM=armhf \
     DOCKER_COMMAND="make -C mediapipe/examples/coral BAZEL_TARGET=mediapipe/examples/coral:face_detection_tpu build" \
     docker
```

and get the output binary from `out/<platform>` directory. Any Mediapipe target
can be cross-compiled  this way, e.g. try
`mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu`.

To summarize everything:

| Arch  | PLATFORM       | Output      | Board                                                    |
| ----- | -------------- | ----------- | -------------------------------------------------------- |
| ARM32 | PLATFORM=armhf | out/armv7a  | [Raspberry Pi](https://www.raspberrypi.org/products/)    |
| ARM64 | PLATFORM=arm64 | out/aarch64 | [Coral Dev Board](https://coral.ai/products/dev-board/)  |

## Coral Examples

There are two Coral examples in `mediapipe/examples/coral` directory. Compile
them for your platform:

```bash
# Face detection
make -C mediapipe/examples/coral \
     PLATFORM=armhf \
     DOCKER_COMMAND="make -C mediapipe/examples/coral BAZEL_TARGET=mediapipe/examples/coral:face_detection_tpu build" \
     docker

# Object detection
make -C mediapipe/examples/coral \
     PLATFORM=armhf \
     DOCKER_COMMAND="make -C mediapipe/examples/coral BAZEL_TARGET=mediapipe/examples/coral:object_detection_tpu build" \
     docker
```

Copy output binaries along with corresponding auxiliary files to your target
system. You can copy the whole `mediapipe` folder for simplicity:

```bash
scp -r mediapipe <user>@<host>:.
```

OpenCV runtime libraries need to be installed on your target system:

```bash
sudo apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev
```

If you are going to connect Coral USB accelerator to your target system then
you'll also need `libusb` library:

```shell
sudo apt-get install -y \
   libusb-1.0-0
```

Connect USB camera and Coral device to your target system and run the copied
binaries:

```bash
# Face Detection
GLOG_logtostderr=1 ./face_detection_tpu --calculator_graph_config_file \
    mediapipe/examples/coral/graphs/face_detection_desktop_live.pbtxt

# Object Detection
GLOG_logtostderr=1 ./object_detection_tpu --calculator_graph_config_file \
    mediapipe/examples/coral/graphs/object_detection_desktop_live.pbtxt
```
