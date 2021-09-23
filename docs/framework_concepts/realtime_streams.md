---
layout: default
title: Real-time Streams
parent: Framework Concepts
nav_order: 6
---

# Real-time Streams
{: .no_toc }

1. TOC
{:toc}
---

## Real-time timestamps

MediaPipe calculator graphs are often used to process streams of video or audio
frames for interactive applications. The MediaPipe framework requires only that
successive packets be assigned monotonically increasing timestamps. By
convention, real-time calculators and graphs use the recording time or the
presentation time of each frame as its timestamp, with each timestamp indicating
the microseconds since `Jan/1/1970:00:00:00`. This allows packets from various
sources to be processed in a globally consistent sequence.

## Real-time scheduling

Normally, each Calculator runs as soon as all of its input packets for a given
timestamp become available. Normally, this happens when the calculator has
finished processing the previous frame, and each of the calculators producing
its inputs have finished processing the current frame. The MediaPipe scheduler
invokes each calculator as soon as these conditions are met. See
[Synchronization](synchronization.md) for more details.

## Timestamp bounds

When a calculator does not produce any output packets for a given timestamp, it
can instead output a "timestamp bound" indicating that no packet will be
produced for that timestamp. This indication is necessary to allow downstream
calculators to run at that timestamp, even though no packet has arrived for
certain streams for that timestamp. This is especially important for real-time
graphs in interactive applications, where it is crucial that each calculator
begin processing as soon as possible.

Consider a graph like the following:

```
node {
   calculator: "A"
   input_stream: "alpha_in"
   output_stream: "alpha"
}
node {
   calculator: "B"
   input_stream: "alpha"
   input_stream: "foo"
   output_stream: "beta"
}
```

Suppose: at timestamp `T`, node `A` doesn't send a packet in its output stream
`alpha`. Node `B` gets a packet in `foo` at timestamp `T` and is waiting for a
packet in `alpha` at timestamp `T`. If `A` doesn't send `B` a timestamp bound
update for `alpha`, `B` will keep waiting for a packet to arrive in `alpha`.
Meanwhile, the packet queue of `foo` will accumulate packets at `T`, `T+1` and
so on.

To output a packet on a stream, a calculator uses the API functions
`CalculatorContext::Outputs` and `OutputStream::Add`. To instead output a
timestamp bound on a stream, a calculator can use the API functions
`CalculatorContext::Outputs` and `CalculatorContext::SetNextTimestampBound`. The
specified bound is the lowest allowable timestamp for the next packet on the
specified output stream. When no packet is output, a calculator will typically
do something like:

```
cc->Outputs().Tag("output_frame").SetNextTimestampBound(
  cc->InputTimestamp().NextAllowedInStream());
```

The function `Timestamp::NextAllowedInStream` returns the successive timestamp.
For example, `Timestamp(1).NextAllowedInStream() == Timestamp(2)`.

## Propagating timestamp bounds

Calculators that will be used in real-time graphs need to define output
timestamp bounds based on input timestamp bounds in order to allow downstream
calculators to be scheduled promptly. A common pattern is for calculators to
output packets with the same timestamps as their input packets. In this case,
simply outputting a packet on every call to `Calculator::Process` is sufficient
to define output timestamp bounds.

However, calculators are not required to follow this common pattern for output
timestamps, they are only required to choose monotonically increasing output
timestamps. As a result, certain calculators must calculate timestamp bounds
explicitly. MediaPipe provides several tools for computing appropriate timestamp
bound for each calculator.

1\. **SetNextTimestampBound()** can be used to specify the timestamp bound, `t +
1`, for an output stream.

```
cc->Outputs.Tag("OUT").SetNextTimestampBound(t.NextAllowedInStream());
```

Alternatively, an empty packet with timestamp `t` can be produced to specify the
timestamp bound `t + 1`.

```
cc->Outputs.Tag("OUT").Add(Packet(), t);
```

The timestamp bound of an input stream is indicated by the packet or the empty
packet on the input stream.

```
Timestamp bound = cc->Inputs().Tag("IN").Value().Timestamp();
```

2\. **TimestampOffset()** can be specified in order to automatically copy the
timestamp bound from input streams to output streams.

```
cc->SetTimestampOffset(0);
```

This setting has the advantage of propagating timestamp bounds automatically,
even when only timestamp bounds arrive and Calculator::Process is not invoked.

3\. **ProcessTimestampBounds()** can be specified in order to invoke
`Calculator::Process` for each new "settled timestamp", where the "settled
timestamp" is the new highest timestamp below the current timestamp bounds.
Without `ProcessTimestampBounds()`, `Calculator::Process` is invoked only with
one or more arriving packets.

```
cc->SetProcessTimestampBounds(true);
```

This setting allows a calculator to perform its own timestamp bounds calculation
and propagation, even when only input timestamps are updated. It can be used to
replicate the effect of `TimestampOffset()`, but it can also be used to
calculate a timestamp bound that takes into account additional factors.

For example, in order to replicate `SetTimestampOffset(0)`, a calculator could
do the following:

```
absl::Status Open(CalculatorContext* cc) {
  cc->SetProcessTimestampBounds(true);
}

absl::Status Process(CalculatorContext* cc) {
  cc->Outputs.Tag("OUT").SetNextTimestampBound(
      cc->InputTimestamp().NextAllowedInStream());
}
```

## Scheduling of Calculator::Open and Calculator::Close

`Calculator::Open` is invoked when all required input side-packets have been
produced. Input side-packets can be provided by the enclosing application or by
"side-packet calculators" inside the graph. Side-packets can be specified from
outside the graph using the API's `CalculatorGraph::Initialize` and
`CalculatorGraph::StartRun`. Side packets can be specified by calculators within
the graph using `CalculatorGraphConfig::OutputSidePackets` and
`OutputSidePacket::Set`.

Calculator::Close is invoked when all of the input streams have become `Done` by
being closed or reaching timestamp bound `Timestamp::Done`.

**Note:** If the graph finishes all pending calculator execution and becomes
`Done`, before some streams become `Done`, then MediaPipe will invoke the
remaining calls to `Calculator::Close`, so that every calculator can produce its
final outputs.

The use of `TimestampOffset` has some implications for `Calculator::Close`. A
calculator specifying `SetTimestampOffset(0)` will by design signal that all of
its output streams have reached `Timestamp::Done` when all of its input streams
have reached `Timestamp::Done`, and therefore no further outputs are possible.
This prevents such a calculator from emitting any packets during
`Calculator::Close`. If a calculator needs to produce a summary packet during
`Calculator::Close`, `Calculator::Process` must specify timestamp bounds such
that at least one timestamp (such as `Timestamp::Max`) remains available during
`Calculator::Close`. This means that such a calculator normally cannot rely upon
`SetTimestampOffset(0)` and must instead specify timestamp bounds explicitly
using `SetNextTimestampBounds()`.
