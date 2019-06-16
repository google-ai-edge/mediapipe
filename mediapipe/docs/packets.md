### Packets

- [Creating a packet](#creating-a-packet)

Each calculator is a node of of a graph. We describe how to create a new calculator, how to initialize a calculator, how to perform its calculations, input and output streams, timestamps, and options

#### Creating a packet
Packets are generally created with `MediaPipe::Adopt()` (from packet.h).

```c++
// Create some data.
auto data = gtl::MakeUnique<MyDataClass>("constructor_argument");
// Create a packet to own the data.
Packet p = Adopt(data.release());
// Make a new packet with the same data and a different timestamp.
Packet p2 = p.At(Timestamp::PostStream());
```

Data within a packet is accessed with `Packet::Get<T>()`
