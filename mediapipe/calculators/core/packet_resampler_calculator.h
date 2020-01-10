#ifndef MEDIAPIPE_CALCULATORS_CORE_PACKET_RESAMPLER_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_PACKET_RESAMPLER_CALCULATOR_H_

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/core/packet_resampler_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/deps/mathutil.h"
#include "mediapipe/framework/deps/random_base.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/options_util.h"

namespace mediapipe {

class PacketReservoir {
 public:
  PacketReservoir(RandomBase* rng) : rng_(rng) {}
  // Replace candidate with current packet with 1/count_ probability.
  void AddSample(Packet sample) {
    if (rng_->UnbiasedUniform(++count_) == 0) {
      reservoir_ = sample;
    }
  }
  bool IsEnabled() { return rng_ && enabled_; }
  void Disable() {
    if (enabled_) enabled_ = false;
  }
  void Clear() { count_ = 0; }
  bool IsEmpty() { return count_ == 0; }
  Packet GetSample() { return reservoir_; }

 private:
  RandomBase* rng_;
  bool enabled_ = true;
  int32 count_ = 0;
  Packet reservoir_;
};

// This calculator is used to normalize the frequency of the packets
// out of a stream. Given a desired frame rate, packets are going to be
// removed or added to achieve it.
//
// The jitter feature is disabled by default. To enable it, you need to
// implement CreateSecureRandom(const std::string&).
//
// The data stream may be either specified as the only stream (by index)
// or as the stream with tag "DATA".
//
// The input and output streams may be accompanied by a VIDEO_HEADER
// stream.  This stream includes a VideoHeader at Timestamp::PreStream().
// The input VideoHeader on the VIDEO_HEADER stream will always be updated
// with the resampler frame rate no matter what the options value for
// output_header is before being output on the output VIDEO_HEADER stream.
// If the input VideoHeader is not available, then only the frame rate
// value will be set in the output.
//
// Related:
//   packet_downsampler_calculator.cc: skips packets regardless of timestamps.
class PacketResamplerCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // Calculates the first sampled timestamp that incorporates a jittering
  // offset.
  void InitializeNextOutputTimestampWithJitter();
  // Calculates the next sampled timestamp that incorporates a jittering offset.
  void UpdateNextOutputTimestampWithJitter();

  // Logic for Process() when jitter_ != 0.0.
  ::mediapipe::Status ProcessWithJitter(CalculatorContext* cc);

  // Logic for Process() when jitter_ == 0.0.
  ::mediapipe::Status ProcessWithoutJitter(CalculatorContext* cc);

  // Given the current count of periods that have passed, this returns
  // the next valid timestamp of the middle point of the next period:
  //    if count is 0, it returns the first_timestamp_.
  //    if count is 1, it returns the first_timestamp_ + period (corresponding
  //       to the first tick using exact fps)
  // e.g. for frame_rate=30 and first_timestamp_=0:
  //    0: 0
  //    1: 33333
  //    2: 66667
  //    3: 100000
  //
  // Can only be used if jitter_ equals zero.
  Timestamp PeriodIndexToTimestamp(int64 index) const;

  // Given a Timestamp, finds the closest sync Timestamp based on
  // first_timestamp_ and the desired fps.
  //
  // Can only be used if jitter_ equals zero.
  int64 TimestampToPeriodIndex(Timestamp timestamp) const;

  // Outputs a packet if it is in range (start_time_, end_time_).
  void OutputWithinLimits(CalculatorContext* cc, const Packet& packet) const;

  // The timestamp of the first packet received.
  Timestamp first_timestamp_;

  // Number of frames per second (desired output frequency).
  double frame_rate_;

  // Inverse of frame_rate_.
  int64 frame_time_usec_;

  // Number of periods that have passed (= #packets sent to the output).
  //
  // Can only be used if jitter_ equals zero.
  int64 period_count_;

  // The last packet that was received.
  Packet last_packet_;

  VideoHeader video_header_;
  // The "DATA" input stream.
  CollectionItemId input_data_id_;
  // The "DATA" output stream.
  CollectionItemId output_data_id_;

  // Indicator whether to flush last packet even if its timestamp is greater
  // than the final stream timestamp.  Set to false when jitter_ is non-zero.
  bool flush_last_packet_;

  // Jitter-related variables.
  std::unique_ptr<RandomBase> random_;
  double jitter_ = 0.0;
  Timestamp next_output_timestamp_;
  Timestamp next_output_timestamp_min_;

  // If specified, output timestamps are aligned with base_timestamp.
  // Otherwise, they are aligned with the first input timestamp.
  Timestamp base_timestamp_;

  // If specified, only outputs at/after start_time are included.
  Timestamp start_time_;

  // If specified, only outputs before end_time are included.
  Timestamp end_time_;

  // If set, the output timestamps nearest to start_time and end_time
  // are included in the output, even if the nearest timestamp is not
  // between start_time and end_time.W
  bool round_limits_;

  // packet reservior used for sampling random packet out of partial
  // period when jitter is enabled
  std::unique_ptr<PacketReservoir> packet_reservoir_;
  // random number generator used in packet_reservior_.
  std::unique_ptr<RandomBase> packet_reservoir_random_;
};

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_CORE_PACKET_RESAMPLER_CALCULATOR_H_
