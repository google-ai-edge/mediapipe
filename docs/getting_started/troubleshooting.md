---
layout: forward
target: https://developers.google.com/mediapipe/framework/getting_started/troubleshooting
title: Troubleshooting
parent: Getting Started
nav_order: 10
---

# Troubleshooting
{: .no_toc }

1. TOC
{:toc}
---

**Attention:** *Thanks for your interest in MediaPipe! We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

--------------------------------------------------------------------------------

## Missing Python binary path

The error message:

```
ERROR: An error occurred during the fetch of repository 'local_execution_config_python':
  Traceback (most recent call last):
       File "/sandbox_path/external/org_tensorflow/third_party/py/python_configure.bzl", line 208
               get_python_bin(repository_ctx)
    ...
Repository command failed
```

usually indicates that Bazel fails to find the local Python binary. To solve
this issue, please first find where the python binary is and then add
`--action_env PYTHON_BIN_PATH=<path to python binary>` to the Bazel command. For
example, you can switch to use the system default python3 binary by the
following command:

```
bazel build -c opt \
  --define MEDIAPIPE_DISABLE_GPU=1 \
  --action_env PYTHON_BIN_PATH=$(which python3) \
  mediapipe/examples/desktop/hello_world
```

## Missing necessary Python packages

The error message:

```
ImportError: No module named numpy
Is numpy installed?
```

usually indicates that certain Python packages are not installed. Please run
`pip install` or `pip3 install` depending on your Python binary version to
install those packages.

## Fail to fetch remote dependency repositories

The error message:

```
ERROR: An error occurred during the fetch of repository 'org_tensorflow':
   java.io.IOException: Error downloading [https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/77e9ffb9b2bfb1a4f7056e62d84039626923e328.tar.gz, https://github.com/tensorflow/tensorflow/archive/77e9ffb9b2bfb1a4f7056e62d84039626923e328.tar.gz] to /sandbox_path/external/org_tensorflow/77e9ffb9b2bfb1a4f7056e62d84039626923e328.tar.gz: Tried to reconnect at offset 9,944,151 but server didn't support it

or

WARNING: Download from https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_swift/releases/download/0.12.1/rules_swift.0.12.1.tar.gz failed: class java.net.ConnectException Connection timed out (Connection timed out)
```

usually indicates that Bazel fails to download necessary dependency repositories
that MediaPipe needs. MediaPipe has several dependency repositories that are
hosted by Google sites. In some regions, you may need to set up a network proxy
or use a VPN to access those resources. You may also need to append
`--host_jvm_args "-DsocksProxyHost=<ip address> -DsocksProxyPort=<port number>"`
to the Bazel command. See
[this GitHub issue](https://github.com/google/mediapipe/issues/581#issuecomment-610356857)
for more details.

If you believe that it's not a network issue, another possibility is that some
resources could be temporarily unavailable, please run `bazel clean --expunge`
and retry it later. If it's still not working, please file a GitHub issue with
the detailed error message.

## Incorrect MediaPipe OpenCV config

The error message:

```
error: undefined reference to 'cv::String::deallocate()'
error: undefined reference to 'cv::String::allocate(unsigned long)'
error: undefined reference to 'cv::VideoCapture::VideoCapture(cv::String const&)'
...
error: undefined reference to 'cv::putText(cv::InputOutputArray const&, cv::String const&, cv::Point, int, double, cv::Scalar, int, int, bool)'
```

usually indicates that OpenCV is not properly configured for MediaPipe. Please
take a look at the "Install OpenCV and FFmpeg" sections in
[Installation](./install.md) to see how to modify MediaPipe's WORKSPACE and
linux_opencv/macos_opencv/windows_opencv.BUILD files for your local opencv
libraries. [This GitHub issue](https://github.com/google/mediapipe/issues/666)
may also help.

## Python pip install failure

The error message:

```
ERROR: Could not find a version that satisfies the requirement mediapipe
ERROR: No matching distribution found for mediapipe
```

after running `pip install mediapipe` usually indicates that there is no
qualified MediaPipe Python for your system. Please note that MediaPipe Python
PyPI officially supports the **64-bit** version of Python 3.7 to 3.10 on the
following OS:

-   x86_64 Linux
-   x86_64 macOS 10.15+
-   amd64 Windows

If the OS is currently supported and you still see this error, please make sure
that both the Python and pip binary are for Python 3.7 to 3.10. Otherwise,
please consider building the MediaPipe Python package locally by following the
instructions [here](python.md#building-mediapipe-python-package).

## Python DLL load failure on Windows

The error message:

```
ImportError: DLL load failed: The specified module could not be found
```

usually indicates that the local Windows system is missing Visual C++
redistributable packages and/or Visual C++ runtime DLLs. This can be solved by
either installing the official
[vc_redist.x64.exe](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)
or installing the "msvc-runtime" Python package by running

```bash
$ python -m pip install msvc-runtime
```

Please note that the "msvc-runtime" Python package is not released or maintained
by Microsoft.

## Native method not found

The error message:

```
java.lang.UnsatisfiedLinkError: No implementation found for void com.google.wick.Wick.nativeWick
```

usually indicates that a needed native library, such as `/libwickjni.so` has not
been loaded or has not been included in the dependencies of the app or cannot be
found for some reason. Note that Java requires every native library to be
explicitly loaded using the function `System.loadLibrary`.

## No registered calculator found

The error message:

```
No registered object with name: OurNewCalculator; Unable to find Calculator "OurNewCalculator"
```

usually indicates that `OurNewCalculator` is referenced by name in a
[`CalculatorGraphConfig`] but that the library target for OurNewCalculator has
not been linked to the application binary. When a new calculator is added to a
calculator graph, that calculator must also be added as a build dependency of
the applications using the calculator graph.

This error is caught at runtime because calculator graphs reference their
calculators by name through the field `CalculatorGraphConfig::Node:calculator`.
When the library for a calculator is linked into an application binary, the
calculator is automatically registered by name through the
[`REGISTER_CALCULATOR`] macro using the [`registration.h`] library. Note that
[`REGISTER_CALCULATOR`] can register a calculator with a namespace prefix,
identical to its C++ namespace. In this case, the calculator graph must also use
the same namespace prefix.

## Out Of Memory error

Exhausting memory can be a symptom of too many packets accumulating inside a
running MediaPipe graph. This can occur for a number of reasons, such as:

1.  Some calculators in the graph simply can't keep pace with the arrival of
    packets from a realtime input stream such as a video camera.
2.  Some calculators are waiting for packets that will never arrive.

For problem (1), it may be necessary to drop some old packets in older to
process the more recent packets. For some hints, see:
[`How to process realtime input streams`].

For problem (2), it could be that one input stream is lacking packets for some
reason. A device or a calculator may be misconfigured or may produce packets
only sporadically. This can cause downstream calculators to wait for many
packets that will never arrive, which in turn causes packets to accumulate on
some of their input streams. MediaPipe addresses this sort of problem using
"timestamp bounds". For some hints see:
[`How to process realtime input streams`].

The MediaPipe setting [`CalculatorGraphConfig::max_queue_size`] limits the
number of packets enqueued on any input stream by throttling inputs to the
graph. For realtime input streams, the number of packets queued at an input
stream should almost always be zero or one. If this is not the case, you may see
the following warning message:

```
Resolved a deadlock by increasing max_queue_size of input stream
```

Also, the setting [`CalculatorGraphConfig::report_deadlock`] can be set to cause
graph run to fail and surface the deadlock as an error, such that max_queue_size
to acts as a memory usage limit.

## Graph hangs

Many applications will call [`CalculatorGraph::CloseAllPacketSources`] and
[`CalculatorGraph::WaitUntilDone`] to finish or suspend execution of a MediaPipe
graph. The objective here is to allow any pending calculators or packets to
complete processing, and then to shutdown the graph. If all goes well, every
stream in the graph will reach [`Timestamp::Done`], and every calculator will
reach [`CalculatorBase::Close`], and then [`CalculatorGraph::WaitUntilDone`]
will complete successfully.

If some calculators or streams cannot reach state [`Timestamp::Done`] or
[`CalculatorBase::Close`], then the method [`CalculatorGraph::Cancel`] can be
called to terminate the graph run without waiting for all pending calculators
and packets to complete.

To understand why a graph hangs/stalls,
[graph runtime monitoring](#graph-runtime-monitoring) can provide valuable
insights into input stream packet queues (understand where packets are queued up
in the MediaPipe graph). This information reveals which calculators are waiting
on their input queues for additional input before triggering their next
Calculator::Process call.

For the specific monitoring of timestamp settlements of a specific calculator in
your MediaPipe graph,
[DebugInputStreamHandler](#monitor-calculator-inputs-and-timestamp-settlements)
can be your friend.

## Output timing is uneven

Some realtime MediaPipe graphs produce a series of video frames for viewing as a
video effect or as a video diagnostic. Sometimes, a MediaPipe graph will produce
these frames in clusters, for example when several output frames are
extrapolated from the same cluster of input frames. If the outputs are presented
as they are produced, some output frames are immediately replaced by later
frames in the same cluster, which makes the results hard to see and evaluate
visually. In cases like this, the output visualization can be improved by
presenting the frames at even intervals in real time.

MediaPipe addresses this use case by mapping timestamps to points in real time.
Each timestamp indicates a time in microseconds, and a calculator such as
`LiveClockSyncCalculator` can delay the output of packets to match their
timestamps. This sort of calculator adjusts the timing of outputs such that:

1.  The time between outputs corresponds to the time between timestamps as
    closely as possible.
2.  Outputs are produced with the smallest delay possible.

## CalculatorGraph lags behind inputs

For many realtime MediaPipe graphs, low latency is an objective. MediaPipe
supports "pipelined" style parallel processing in order to begin processing of
each packet as early as possible. Normally the lowest possible latency is the
total time required by each calculator along a "critical path" of successive
calculators. The latency of the a MediaPipe graph could be worse than the ideal
due to delays introduced to display frames a even intervals as described in
[Output timing is uneven](#output-timing-is-uneven).

If some of the calculators in the graph cannot keep pace with the realtime input
streams, then latency will continue to increase, and it becomes necessary to
drop some input packets. The recommended technique is to use the MediaPipe
calculators designed specifically for this purpose such as
[`FlowLimiterCalculator`] as described in
[`How to process realtime input streams`].

## Monitor calculator inputs and timestamp settlements

Debugging MediaPipe calculators often requires a deep understanding of the data
flow and timestamp synchronization. Incoming packets to calculators are first
buffered in input queues per stream to be synchronized by the assigned
`InputStreamHandler`. The `InputStreamHandler` job is to determine the input
packet set for a settled timestamp, which puts the calculator into a “ready”
state, followed by triggering a Calculator::Process call with the determined
packet set as input.

The `DebugInputStreamHandler` can be used to track incoming packets and
timestamp settlements in real-time in the application's LOG(INFO) output. It can
be assigned to specific calculators via the Calculator's input_stream_handler or
graph globally via the `CalculatorGraphConfig`'s input_stream_handler field.

During the graph execution, incoming packets generate LOG messages which reveal
the timestamp and type of the packet, followed by the current state of all input
queues:

```
[INFO] SomeCalculator: Adding packet (ts:2, type:int) to stream INPUT_B:0:input_b
[INFO] SomeCalculator: INPUT_A:0:input_a num_packets: 0 min_ts: 2
[INFO] SomeCalculator: INPUT_B:0:input_b num_packets: 1 min_ts: 2
```

In addition, it enables the monitoring of timestamp settlement events (in case
the `DefaultInputStreamHandler` is applied). This can help to reveal an
unexpected timestamp bound increase on input streams resulting in a
Calculator::Process call with an incomplete input set resulting in empty packets
on (potentially required) input streams.

*Example scenario:*

```
node {
  calculator: "SomeCalculator"
  input_stream: "INPUT_A:a"
  input_stream: "INPUT_B:b"
  ...
}
```

Given a calculator with two inputs, receiving an incoming packet with timestamp
1 on stream A followed by an input packet with timestamp 2 on stream B. The
timestamp bound increase to 2 on stream B with pending input packet on stream A
at timestamp 1 triggers the Calculator::Process call with an incomplete input
set for timestamp 1. In this case, the `DefaultInputStreamHandler` outputs:

```
[INFO] SomeCalculator: Filled input set at ts: 1 with MISSING packets in input streams: INPUT_B:0:input_b.
```

## Graph runtime monitoring

Graph runtime monitoring can be a helpful tool to debug stalled MediaPipe
graphs. It utilizes a background thread to periodically capture a "snapshot" of
the graph's calculators and input/output streams state at predetermined
intervals.

The output begins by listing the calculators that are currently running
(Calculator::Process runs),

```
Running calculators: PacketClonerCalculator, RectTransformationCalculator
```

followed by an overview of packets that are currently in flight, as well as
those waiting in calculator input streams/queues to be processed.

```
Running calculators: PacketClonerCalculator
Num packets in input queues: 4
GateCalculator_2 waiting on stream(s): :1:norm_start_rect
MergeCalculator waiting on stream(s): :0:output_frames_gpu_ao, :1:segmentation_preview_gpu
```

The monitoring output continues with a detailed overview of all calculator
states, including timestamp bounds, time of last activity, and statistics about
their input and output streams.

```
PreviousLoopbackCalculator: (idle for 8.17s, ts bound : 0)
Input streams:
 * LOOP:0:segmentation_finished - queue size: 0, total added: 0, ts bound: 569604400011
 * MAIN:0:input_frames_gpu - queue size: 0, total added: 2, ts bound: 569604400011
Output streams:
 * PREV_LOOP:0:prev_segmentation_finished, total added: 0, ts bound: 569604400011
```

Graph runtime monitoring can be enabled with the flag
`enable_graph_runtime_info`. This enables the background capturing of graph
runtime monitoring which is written to LOG(INFO).

```
graph {
  runtime_info {
    enable_graph_runtime_info: true
  }
  ...
}
```

Since adb logging might be throttled for larger graph runtime information, as an
alternative, its output can be written to a file at the specified capture
interval. This will overwrite the file each time. To enable this, use the flag
`mp_graph_runtime_info_output_file`. Note: On Android, the output file may need
to be created first to avoid permission issues.

## VLOG is your friend

MediaPipe uses `VLOG` in many places to log important events for debugging
purposes, while not affecting performance if logging is not enabled.

See more about `VLOG` on [abseil `VLOG`]

Mind that `VLOG` can be spammy if you enable it globally e.g. (using `--v`
flag). The solution `--vmodule` flag that allows different levels to be set for
different source files.

In cases when `--v` / `--vmodule` cannot be used (e.g. running an Android app),
MediaPipe allows to set `VLOG` `--v` / `--vmodule` flags overrides for debugging
purposes which are applied when `CalculatorGraph` is created.

Overrides:

-   `MEDIAPIPE_VLOG_V`: define and provide value you provide for `--v`
-   `MEDIAPIPE_VLOG_VMODULE`: define and provide value you provide for
    `--vmodule`

You can set overrides by adding:
`--copt=-DMEDIAPIPE_VLOG_VMODULE=\"*calculator*=5\"`

with your desired module patterns and `VLOG` levels (see more details for
`--vmodule` at [abseil `VLOG`]) to your build command.

IMPORTANT: mind that adding the above to your build command will trigger rebuild
of the whole binary including dependencies. So, considering `VLOG` overrides
exist for debugging purposes only, it is faster to simply modify
[`vlog_overrides.cc`] adding `MEDIAPIPE_VLOG_V/VMODULE` at the very top.

[`CalculatorGraphConfig`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator.proto
[`CalculatorGraphConfig::max_queue_size`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator.proto
[`CalculatorGraphConfig::report_deadlock`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator.proto
[`REGISTER_CALCULATOR`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator_registry.h
[`registration.h`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/deps/registration.h
[`CalculatorGraph::CloseAllPacketSources`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator_graph.h
[`CalculatorGraph::Cancel`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator_graph.h
[`CalculatorGraph::WaitUntilDone`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator_graph.h
[`Timestamp::Done`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/timestamp.h
[`CalculatorBase::Close`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/calculator_base.h
[`FlowLimiterCalculator`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/calculators/core/flow_limiter_calculator.cc
[`How to process realtime input streams`]: faq.md#how-to-process-realtime-input-streams
[`vlog_overrides.cc`]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/framework/vlog_overrides.cc
[abseil `VLOG`]: https://abseil.io/docs/cpp/guides/logging#VLOG

## Unsupported flags during build

If you are using Clang 18 or older, you may have to disable some compiler
optimizations in our CPU backend.

To disable support for `avxvnniint8`, add the following to you `.bazelrc`:

```
build --define=xnn_enable_avxvnniint8=false
```
