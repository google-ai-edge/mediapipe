## Building MediaPipe Calculators

-   [Example calculator](#example-calculator)


### Example calculator

This section discusses the implementation of `PacketClonerCalculator`, which
does a relatively simple job, and is used in many calculator graphs.
`PacketClonerCalculator` simply produces a copy of its most recent input
packets on demand.

`PacketClonerCalculator` is useful when the timestamps of arriving data packets
are not aligned perfectly. Suppose we have a room with a microphone, light
sensor and a video camera that is collecting sensory data. Each of the sensors
operates independently and collects data intermittently. Suppose that the output
of each sensor is:

*   microphone = loudness in decibels of sound in the room (Integer)
*   light sensor = brightness of room (Integer)
*   video camera = RGB image frame of room (ImageFrame)

Our simple perception pipeline is designed to process sensory data from these 3
sensors such that at any time when we have image frame data from the camera that
is synchronized with the last collected microphone loudness data and light
sensor brightness data. To do this with MediaPipe, our perception pipeline has 3
input streams:

*   room_mic_signal - Each packet of data in this input stream is integer data
    representing how loud audio is in a room with timestamp.
*   room_lightening_sensor - Each packet of data in this input stream is integer
    data representing how bright is the room illuminated with timestamp.
*   room_video_tick_signal - Each packet of data in this input stream is
    imageframe of video data representing video collected from camera in the
    room with timestamp.

Below is the implementation of the `PacketClonerCalculator`.  You can see
the `GetContract()`, `Open()`, and `Process()` methods as well as the instance
variable `current_` which holds the most recent input packets.

```c++
// This takes packets from N+1 streams, A_1, A_2, ..., A_N, B.
// For every packet that appears in B, outputs the most recent packet from each
// of the A_i on a separate stream.

#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

// For every packet received on the last stream, output the latest packet
// obtained on all other streams. Therefore, if the last stream outputs at a
// higher rate than the others, this effectively clones the packets from the
// other streams to match the last.
//
// Example config:
// node {
//   calculator: "PacketClonerCalculator"
//   input_stream: "first_base_signal"
//   input_stream: "second_base_signal"
//   input_stream: "tick_signal"
//   output_stream: "cloned_first_base_signal"
//   output_stream: "cloned_second_base_signal"
// }
//
class PacketClonerCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    const int tick_signal_index = cc->Inputs().NumEntries() - 1;
    // cc->Inputs().NumEntries() returns the number of input streams
    // for the PacketClonerCalculator
    for (int i = 0; i < tick_signal_index; ++i) {
      cc->Inputs().Index(i).SetAny();
      // cc->Inputs().Index(i) returns the input stream pointer by index
      cc->Outputs().Index(i).SetSameAs(&cc->Inputs().Index(i));
    }
    cc->Inputs().Index(tick_signal_index).SetAny();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    tick_signal_index_ = cc->Inputs().NumEntries() - 1;
    current_.resize(tick_signal_index_);
    // Pass along the header for each stream if present.
    for (int i = 0; i < tick_signal_index_; ++i) {
      if (!cc->Inputs().Index(i).Header().IsEmpty()) {
        cc->Outputs().Index(i).SetHeader(cc->Inputs().Index(i).Header());
        // Sets the output stream of index i header to be the same as
        // the header for the input stream of index i
      }
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    // Store input signals.
    for (int i = 0; i < tick_signal_index_; ++i) {
      if (!cc->Inputs().Index(i).Value().IsEmpty()) {
        current_[i] = cc->Inputs().Index(i).Value();
      }
    }

    // Output if the tick signal is non-empty.
    if (!cc->Inputs().Index(tick_signal_index_).Value().IsEmpty()) {
      for (int i = 0; i < tick_signal_index_; ++i) {
        if (!current_[i].IsEmpty()) {
          cc->Outputs().Index(i).AddPacket(
              current_[i].At(cc->InputTimestamp()));
          // Add a packet to output stream of index i a packet from inputstream i
          // with timestamp common to all present inputs
          //
        } else {
          cc->Outputs().Index(i).SetNextTimestampBound(
              cc->InputTimestamp().NextAllowedInStream());
          // if current_[i], 1 packet buffer for input stream i is empty, we will set
          // next allowed timestamp for input stream i to be current timestamp + 1
        }
      }
    }
    return ::mediapipe::OkStatus();
  }

 private:
  std::vector<Packet> current_;
  int tick_signal_index_;
};

REGISTER_CALCULATOR(PacketClonerCalculator);
}  // namespace mediapipe
```

Typically, a calculator has only a .cc file. No .h is required, because
mediapipe uses registration to make calculators known to it. After you have
defined your calculator class, register it with a macro invocation
REGISTER_CALCULATOR(calculator_class_name).

Below is a trivial MediaPipe graph that has 3 input streams, 1 node
(PacketClonerCalculator) and 3 output streams.

```proto
input_stream: "room_mic_signal"
input_stream: "room_lighting_sensor"
input_stream: "room_video_tick_signal"

node {
   calculator: "PacketClonerCalculator"
   input_stream: "room_mic_signal"
   input_stream: "room_lighting_sensor"
   input_stream: "room_video_tick_signal"
   output_stream: "cloned_room_mic_signal"
   output_stream: "cloned_lighting_sensor"
 }
```

The diagram below shows how the `PacketClonerCalculator` defines its output
packets based on its series of input packets.

| ![Graph using PacketClonerCalculator](images/packet_cloner_calculator.png) |
|:--:|
| *Each time it receives a packet on its TICK input stream, the PacketClonerCalculator outputs the most recent packet from each of its input streams.  The sequence of output packets is determined by the sequene of input packets and their timestamps. The timestamps are shows along the right side of the diagram.* |


