---
layout: default
title: GPU Support
parent: Getting Started
nav_order: 7
---

# GPU Support
{: .no_toc }

1. TOC
{:toc}
---

## OpenGL ES Support

MediaPipe supports OpenGL ES up to version 3.2 on Android/Linux and up to ES 3.0
on iOS. In addition, MediaPipe also supports Metal on iOS.

OpenGL ES 3.1 or greater is required (on Android/Linux systems) for running
machine learning inference calculators and graphs.

## Disable OpenGL ES Support

By default, building MediaPipe (with no special bazel flags) attempts to compile
and link against OpenGL ES (and for iOS also Metal) libraries.

On platforms where OpenGL ES is not available (see also
[OpenGL ES Setup on Linux Desktop](#opengl-es-setup-on-linux-desktop)), you
should disable OpenGL ES support with:

```
$ bazel build --define MEDIAPIPE_DISABLE_GPU=1 <my-target>
```

Note: On Android and iOS, OpenGL ES is required by MediaPipe framework and the
support should never be disabled.

## OpenGL ES Setup on Linux Desktop

On Linux desktop with video cards that support OpenGL ES 3.1+, MediaPipe can run
GPU compute and rendering and perform TFLite inference on GPU.

To check if your Linux desktop GPU can run MediaPipe with OpenGL ES:

```bash
$ sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
$ sudo apt-get install mesa-utils
$ glxinfo | grep -i opengl
```

For example, it may print:

```bash
$ glxinfo | grep -i opengl
...
OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 430.50
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.20
OpenGL ES profile extensions:
```

If you have connected to your computer through SSH and find when you probe for
GPU information you see the output:

```bash
glxinfo | grep -i opengl
Error: unable to open display
```

Try re-establishing your SSH connection with the `-X` option and try again. For
example:

```bash
ssh -X <user>@<host>
```

*Notice the ES 3.20 text above.*

You need to see ES 3.1 or greater printed in order to perform TFLite inference
on GPU in MediaPipe. With this setup, build with:

```
$ bazel build --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 <my-target>
```

If only ES 3.0 or below is supported, you can still build MediaPipe targets that
don't require TFLite inference on GPU with:

```
$ bazel build --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --copt -DMEDIAPIPE_DISABLE_GL_COMPUTE <my-target>
```

Note: MEDIAPIPE_DISABLE_GL_COMPUTE is already defined automatically on all Apple
systems (Apple doesn't support OpenGL ES 3.1+).

## TensorFlow CUDA Support and Setup on Linux Desktop

MediaPipe framework doesn't require CUDA for GPU compute and rendering. However,
MediaPipe can work with TensorFlow to perform GPU inference on video cards that
support CUDA.

To enable TensorFlow GPU inference with MediaPipe, the first step is to follow
the
[TensorFlow GPU documentation](https://www.tensorflow.org/install/gpu#software_requirements)
to install the required NVIDIA software on your Linux desktop.

After installation, update `$PATH` and `$LD_LIBRARY_PATH` and run `ldconfig`
with:

```
$ export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64,/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
$ sudo ldconfig
```

It's recommended to verify the installation of CUPTI, CUDA, CuDNN, and NVCC:

```
$ ls /usr/local/cuda/extras/CUPTI
/lib64
libcupti.so       libcupti.so.10.1.208  libnvperf_host.so        libnvperf_target.so
libcupti.so.10.1  libcupti_static.a     libnvperf_host_static.a

$ ls /usr/local/cuda-10.1
LICENSE  bin  extras   lib64      libnvvp           nvml  samples  src      tools
README   doc  include  libnsight  nsightee_plugins  nvvm  share    targets  version.txt

$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243

$ ls /usr/lib/x86_64-linux-gnu/ | grep libcudnn.so
libcudnn.so
libcudnn.so.7
libcudnn.so.7.6.4
```

Setting `$TF_CUDA_PATHS` is the way to declare where the CUDA library is. Note
that the following code snippet also adds `/usr/lib/x86_64-linux-gnu` and
`/usr/include` into `$TF_CUDA_PATHS` for cudablas and libcudnn.

```
$ export TF_CUDA_PATHS=/usr/local/cuda-10.1,/usr/lib/x86_64-linux-gnu,/usr/include
```

To make MediaPipe get TensorFlow's CUDA settings, find TensorFlow's
[.bazelrc](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc) and
copy the `build:using_cuda` and `build:cuda` section into MediaPipe's .bazelrc
file. For example, as of April 23, 2020, TensorFlow's CUDA setting is the
following:

```
# This config refers to building with CUDA available. It does not necessarily
# mean that we build CUDA op kernels.
build:using_cuda --define=using_cuda=true
build:using_cuda --action_env TF_NEED_CUDA=1
build:using_cuda --crosstool_top=@local_config_cuda//crosstool:toolchain

# This config refers to building CUDA op kernels with nvcc.
build:cuda --config=using_cuda
build:cuda --define=using_cuda_nvcc=true
```

Finally, build MediaPipe with TensorFlow GPU with two more flags `--config=cuda`
and `--spawn_strategy=local`. For example:

```
$ bazel build -c opt --config=cuda --spawn_strategy=local \
    --define no_aws_support=true --copt -DMESA_EGL_NO_X11_HEADERS \
    mediapipe/examples/desktop/object_detection:object_detection_tensorflow
```

While the binary is running, it prints out the GPU device info:

```
I external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
I external/org_tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc:1544] Found device 0 with properties: pciBusID: 0000:00:04.0 name: Tesla T4 computeCapability: 7.5 coreClock: 1.59GHz coreCount: 40 deviceMemorySize: 14.75GiB deviceMemoryBandwidth: 298.08GiB/s
I external/org_tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc:1686] Adding visible gpu devices: 0
```

You can monitor the GPU usage to verify whether the GPU is used for model
inference.

```
$ nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1

0 %
0 %
4 %
5 %
83 %
21 %
22 %
27 %
29 %
100 %
0 %
0%
```
