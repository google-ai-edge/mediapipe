---
layout: forward
target: https://developers.google.com/mediapipe/framework/getting_started/faq
title: FAQ
parent: Getting Started
nav_order: 9
---

# FAQ
{: .no_toc }

1. TOC
{:toc}
---

**Attention:** *Thanks for your interest in MediaPipe! We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

----

### How to convert ImageFrames and GpuBuffers

The Calculators [`ImageFrameToGpuBufferCalculator`] and
[`GpuBufferToImageFrameCalculator`] convert back and forth between packets of
type [`ImageFrame`] and [`GpuBuffer`]. [`ImageFrame`] refers to image data in
CPU memory in any of a number of bitmap image formats. [`GpuBuffer`] refers to
image data in GPU memory. You can find more detail in the Framework Concepts
section
[GpuBuffer to ImageFrame Converters](./gpu.md#gpubuffer-to-imageframe-converters).
You can see an example in:

*   [`object_detection_mobile_cpu.pbtxt`]

### How to visualize perception results

The [`AnnotationOverlayCalculator`] allows perception results, such as bounding
boxes, arrows, and ovals, to be superimposed on the video frames aligned with
the recognized objects. The results can be displayed in a diagnostic window when
running on a workstation, or in a texture frame when running on device. You can
see an example use of [`AnnotationOverlayCalculator`] in:

*   [`face_detection_mobile_gpu.pbtxt`].

### How to run calculators in parallel

Within a calculator graph, MediaPipe routinely runs separate calculator nodes
in parallel.  MediaPipe maintains a pool of threads, and runs each calculator
as soon as a thread is available and all of it's inputs are ready.  Each
calculator instance is only run for one set of inputs at a time, so most
calculators need only to be *thread-compatible* and not *thread-safe*.

In order to enable one calculator to process multiple inputs in parallel, there
are two possible approaches:

1.  Define multiple calculator nodes and dispatch input packets to all nodes.
2.  Make the calculator thread-safe and configure its [`max_in_flight`] setting.

The first approach can be followed using the calculators designed to distribute
packets across other calculators, such as [`RoundRobinDemuxCalculator`]. A
single [`RoundRobinDemuxCalculator`] can distribute successive packets across
several identically configured [`ScaleImageCalculator`] nodes.

The second approach allows up to [`max_in_flight`] invocations of the
[`CalculatorBase::Process`] method on the same calculator node. The output
packets from [`CalculatorBase::Process`] are automatically ordered by timestamp
before they are passed along to downstream calculators.

With either approach, you must be aware that the calculator running in parallel
cannot maintain internal state in the same way as a normal sequential
calculator.

### Output timestamps when using ImmediateInputStreamHandler

The [`ImmediateInputStreamHandler`] delivers each packet as soon as it arrives
at an input stream. As a result, it can deliver a packet
with a higher timestamp from one input stream before delivering a packet with a
lower timestamp from a different input stream. If these input timestamps are
both used for packets sent to one output stream, that output stream will
complain that the timestamps are not monotonically increasing. In order to
remedy this, the calculator must take care to output a packet only after
processing is complete for its timestamp. This could be accomplished by waiting
until input packets have been received from all inputstreams for that timestamp,
or by ignoring a packet that arrives with a timestamp that has already been
processed.

### How to change settings at runtime

There are two main approaches to changing the settings of a calculator graph
while the application is running:

1. Restart the calculator graph with modified [`CalculatorGraphConfig`].
2. Send new calculator options through packets on graph input-streams.

The first approach has the advantage of leveraging [`CalculatorGraphConfig`]
processing tools such as "subgraphs". The second approach has the advantage of
allowing active calculators and packets to remain in-flight while settings
change. MediaPipe contributors are currently investigating alternative approaches
to achieve both of these advantages.

### How to process realtime input streams

The MediaPipe framework can be used to process data streams either online or
offline. For offline processing, packets are pushed into the graph as soon as
calculators are ready to process those packets. For online processing, one
packet for each frame is pushed into the graph as that frame is recorded.

The MediaPipe framework requires only that successive packets be assigned
monotonically increasing timestamps. By convention, realtime calculators and
graphs use the recording time or the presentation time as the timestamp for each
packet, with each timestamp representing microseconds since
`Jan/1/1970:00:00:00`. This allows packets from various sources to be processed
in a globally consistent order.

Normally for offline processing, every input packet is processed and processing
continues as long as necessary. For online processing, it is often necessary to
drop input packets in order to keep pace with the arrival of input data frames.
When inputs arrive too frequently, the recommended technique for dropping
packets is to use the MediaPipe calculators designed specifically for this
purpose such as [`FlowLimiterCalculator`] and [`PacketClonerCalculator`].

For online processing, it is also necessary to promptly determine when processing
can proceed. MediaPipe supports this by propagating timestamp bounds between
calculators. Timestamp bounds indicate timestamp intervals that will contain no
input packets, and they allow calculators to begin processing for those
timestamps immediately. Calculators designed for realtime processing should
carefully calculate timestamp bounds in order to begin processing as promptly as
possible. For example, the [`MakePairCalculator`] uses the `SetOffset` API to
propagate timestamp bounds from input streams to output streams.

### Can I run MediaPipe on MS Windows?

Currently MediaPipe portability supports Debian Linux, Ubuntu Linux,
MacOS, Android, and iOS.  The core of MediaPipe framework is a C++ library
conforming to the C++11 standard, so it is relatively easy to port to
additional platforms.

[`object_detection_mobile_cpu.pbtxt`]: https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_mobile_cpu.pbtxt
[`ImageFrame`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/formats/image_frame.h
[`GpuBuffer`]: https://github.com/google/mediapipe/tree/master/mediapipe/gpu/gpu_buffer.h
[`GpuBufferToImageFrameCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/gpu/gpu_buffer_to_image_frame_calculator.cc
[`ImageFrameToGpuBufferCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/gpu/image_frame_to_gpu_buffer_calculator.cc
[`AnnotationOverlayCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/calculators/util/annotation_overlay_calculator.cc
[`face_detection_mobile_gpu.pbtxt`]: https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
[`CalculatorBase::Process`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/calculator_base.h
[`max_in_flight`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/calculator.proto
[`RoundRobinDemuxCalculator`]: https://github.com/google/mediapipe/tree/master//mediapipe/calculators/core/round_robin_demux_calculator.cc
[`ScaleImageCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/calculators/image/scale_image_calculator.cc
[`ImmediateInputStreamHandler`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/stream_handler/immediate_input_stream_handler.cc
[`CalculatorGraphConfig`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/calculator.proto
[`FlowLimiterCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/calculators/core/flow_limiter_calculator.cc
[`PacketClonerCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/calculators/core/packet_cloner_calculator.cc
[`MakePairCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/calculators/core/make_pair_calculator.cc
