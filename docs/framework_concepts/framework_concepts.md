---
layout: forward
target: https://developers.google.com/mediapipe/framework/framework_concepts/overview
title: Framework Concepts
nav_order: 5
has_children: true
has_toc: false
---

# Framework Concepts
{: .no_toc }

1. TOC
{:toc}
---

**Attention:** *Thanks for your interest in MediaPipe! We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

----

## The basics

### Packet

The basic data flow unit. A packet consists of a numeric timestamp and a shared
pointer to an **immutable** payload. The payload can be of any C++ type, and the
payload's type is also referred to as the type of the packet. Packets are value
classes and can be copied cheaply. Each copy shares ownership of the payload,
with reference-counting semantics. Each copy has its own timestamp. See also
[Packet](packets.md).

### Graph

MediaPipe processing takes place inside a graph, which defines packet flow paths
between **nodes**. A graph can have any number of inputs and outputs, and data
flow can branch and merge. Generally data flows forward, but backward loops are
possible. See [Graphs](graphs.md) for details.

### Nodes

Nodes produce and/or consume packets, and they are where the bulk of the graph’s
work takes place. They are also known as “calculators”, for historical reasons.
Each node’s interface defines a number of input and output **ports**, identified
by a tag and/or an index. See [Calculators](calculators.md) for details.

### Streams

A stream is a connection between two nodes that carries a sequence of packets,
whose timestamps must be monotonically increasing.

### Side packets

A side packet connection between nodes carries a single packet (with unspecified
timestamp). It can be used to provide some data that will remain constant,
whereas a stream represents a flow of data that changes over time.

### Packet Ports

A port has an associated type; packets transiting through the port must be of
that type. An output stream port can be connected to any number of input stream
ports of the same type; each consumer receives a separate copy of the output
packets, and has its own queue, so it can consume them at its own pace.
Similarly, a side packet output port can be connected to as many side packet
input ports as desired.

A port can be required, meaning that a connection must be made for the graph to
be valid, or optional, meaning it may remain unconnected.

Note: even if a stream connection is required, the stream may not carry a packet
for all timestamps.

## Input and output

Data flow can originate from **source nodes**, which have no input streams and
produce packets spontaneously (e.g. by reading from a file); or from **graph
input streams**, which let an application feed packets into a graph.

Similarly, there are **sink nodes** that receive data and write it to various
destinations (e.g. a file, a memory buffer, etc.), and an application can also
receive output from the graph using **callbacks**.

## Runtime behavior

### Graph lifetime

Once a graph has been initialized, it can be **started** to begin processing
data, and can process a stream of packets until each stream is closed or the
graph is **canceled**. Then the graph can be destroyed or **started** again.

### Node lifetime

There are three main lifetime methods the framework will call on a node:

-   Open: called once, before the other methods. When it is called, all input
    side packets required by the node will be available.
-   Process: called multiple times, when a new set of inputs is available,
    according to the node’s input policy.
-   Close: called once, at the end.

In addition, each calculator can define constructor and destructor, which are
useful for creating and deallocating resources that are independent of the
processed data.

### Input policies

The default input policy is deterministic collation of packets by timestamp. A
node receives all inputs for the same timestamp at the same time, in an
invocation of its Process method; and successive input sets are received in
their timestamp order. This can require delaying the processing of some packets
until a packet with the same timestamp is received on all input streams, or
until it can be guaranteed that a packet with that timestamp will not be
arriving on the streams that have not received it.

Other policies are also available, implemented using a separate kind of
component known as an InputStreamHandler.

See [Synchronization](synchronization.md) for more details.

### Real-time streams

MediaPipe calculator graphs are often used to process streams of video or audio
frames for interactive applications. Normally, each Calculator runs as soon as
all of its input packets for a given timestamp become available. Calculators
used in real-time graphs need to define output timestamp bounds based on input
timestamp bounds in order to allow downstream calculators to be scheduled
promptly. See [Real-time Streams](realtime_streams.md) for details.
