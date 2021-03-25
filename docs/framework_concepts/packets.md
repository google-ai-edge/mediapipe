---
layout: default
title: Packets
parent: Framework Concepts
nav_order: 3
---

# Packets
{: .no_toc }

1. TOC
{:toc}
---

Calculators communicate by sending and receiving packets. Typically a single
packet is sent along each input stream at each input timestamp. A packet can
contain any kind of data, such as a single frame of video or a single integer
detection count.

## Creating a packet

Packets are generally created with `mediapipe::MakePacket<T>()` or
`mediapipe::Adopt()` (from packet.h).

```c++
// Create a packet containing some new data.
Packet p = MakePacket<MyDataClass>("constructor_argument");
// Make a new packet with the same data and a different timestamp.
Packet p2 = p.At(Timestamp::PostStream());
```

or:

```c++
// Create some new data.
auto data = absl::make_unique<MyDataClass>("constructor_argument");
// Create a packet to own the data.
Packet p = Adopt(data.release()).At(Timestamp::PostStream());
```

Data within a packet is accessed with `Packet::Get<T>()`
