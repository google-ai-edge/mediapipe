// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

namespace {

// Creates a secure random number generator for use in ProcessWithJitter.
// If no secure random number generator can be constructed, the jitter
// option is disabled in order to mainatain a consistent security and
// consistent random seeding.
std::unique_ptr<RandomBase> CreateSecureRandom(const std::string& seed) {
  RandomBase* result = nullptr;
  return std::unique_ptr<RandomBase>(result);
}

}  // namespace

namespace mediapipe {

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

  // If specified, output timestamps are aligned with base_timestamp.
  // Otherwise, they are aligned with the first input timestamp.
  Timestamp base_timestamp_;

  // If specified, only outputs at/after start_time are included.
  Timestamp start_time_;

  // If specified, only outputs before end_time are included.
  Timestamp end_time_;

  // If set, the output timestamps nearest to start_time and end_time
  // are included in the output, even if the nearest timestamp is not
  // between start_time and end_time.
  bool round_limits_;
};

REGISTER_CALCULATOR(PacketResamplerCalculator);

namespace {
// Returns a TimestampDiff (assuming microseconds) corresponding to the
// given time in seconds.
TimestampDiff TimestampDiffFromSeconds(double seconds) {
  return TimestampDiff(MathUtil::SafeRound<int64, double>(
      seconds * Timestamp::kTimestampUnitsPerSecond));
}
}  // namespace

::mediapipe::Status PacketResamplerCalculator::GetContract(
    CalculatorContract* cc) {
  const auto& resampler_options =
      cc->Options<PacketResamplerCalculatorOptions>();
  if (cc->InputSidePackets().HasTag("OPTIONS")) {
    cc->InputSidePackets().Tag("OPTIONS").Set<CalculatorOptions>();
  }
  CollectionItemId input_data_id = cc->Inputs().GetId("DATA", 0);
  if (!input_data_id.IsValid()) {
    input_data_id = cc->Inputs().GetId("", 0);
  }
  cc->Inputs().Get(input_data_id).SetAny();
  if (cc->Inputs().HasTag("VIDEO_HEADER")) {
    cc->Inputs().Tag("VIDEO_HEADER").Set<VideoHeader>();
  }

  CollectionItemId output_data_id = cc->Outputs().GetId("DATA", 0);
  if (!output_data_id.IsValid()) {
    output_data_id = cc->Outputs().GetId("", 0);
  }
  cc->Outputs().Get(output_data_id).SetSameAs(&cc->Inputs().Get(input_data_id));
  if (cc->Outputs().HasTag("VIDEO_HEADER")) {
    cc->Outputs().Tag("VIDEO_HEADER").Set<VideoHeader>();
  }

  if (resampler_options.jitter() != 0.0) {
    RET_CHECK_GT(resampler_options.jitter(), 0.0);
    RET_CHECK_LE(resampler_options.jitter(), 1.0);
    RET_CHECK(cc->InputSidePackets().HasTag("SEED"));
    cc->InputSidePackets().Tag("SEED").Set<std::string>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketResamplerCalculator::Open(CalculatorContext* cc) {
  const auto resampler_options =
      tool::RetrieveOptions(cc->Options<PacketResamplerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");

  flush_last_packet_ = resampler_options.flush_last_packet();
  jitter_ = resampler_options.jitter();

  input_data_id_ = cc->Inputs().GetId("DATA", 0);
  if (!input_data_id_.IsValid()) {
    input_data_id_ = cc->Inputs().GetId("", 0);
  }
  output_data_id_ = cc->Outputs().GetId("DATA", 0);
  if (!output_data_id_.IsValid()) {
    output_data_id_ = cc->Outputs().GetId("", 0);
  }

  period_count_ = 0;
  frame_rate_ = resampler_options.frame_rate();
  base_timestamp_ = resampler_options.has_base_timestamp()
                        ? Timestamp(resampler_options.base_timestamp())
                        : Timestamp::Unset();
  start_time_ = resampler_options.has_start_time()
                    ? Timestamp(resampler_options.start_time())
                    : Timestamp::Min();
  end_time_ = resampler_options.has_end_time()
                  ? Timestamp(resampler_options.end_time())
                  : Timestamp::Max();
  round_limits_ = resampler_options.round_limits();
  // The frame_rate has a default value of -1.0, so the user must set it!
  RET_CHECK_LT(0, frame_rate_)
      << "The output frame rate must be greater than zero";
  RET_CHECK_LE(frame_rate_, Timestamp::kTimestampUnitsPerSecond)
      << "The output frame rate must be smaller than "
      << Timestamp::kTimestampUnitsPerSecond;

  frame_time_usec_ = static_cast<int64>(1000000.0 / frame_rate_);
  video_header_.frame_rate = frame_rate_;

  if (resampler_options.output_header() !=
          PacketResamplerCalculatorOptions::NONE &&
      !cc->Inputs().Get(input_data_id_).Header().IsEmpty()) {
    if (resampler_options.output_header() ==
        PacketResamplerCalculatorOptions::UPDATE_VIDEO_HEADER) {
      video_header_ =
          cc->Inputs().Get(input_data_id_).Header().Get<VideoHeader>();
      video_header_.frame_rate = frame_rate_;
      cc->Outputs()
          .Get(output_data_id_)
          .SetHeader(Adopt(new VideoHeader(video_header_)));
    } else {
      cc->Outputs()
          .Get(output_data_id_)
          .SetHeader(cc->Inputs().Get(input_data_id_).Header());
    }
  }

  if (jitter_ != 0.0) {
    if (resampler_options.output_header() !=
        PacketResamplerCalculatorOptions::NONE) {
      LOG(WARNING) << "VideoHeader::frame_rate holds the target value and not "
                      "the actual value.";
    }
    if (flush_last_packet_) {
      flush_last_packet_ = false;
      LOG(WARNING) << "PacketResamplerCalculatorOptions.flush_last_packet is "
                      "ignored, because we are adding jitter.";
    }
    const auto& seed = cc->InputSidePackets().Tag("SEED").Get<std::string>();
    random_ = CreateSecureRandom(seed);
    if (random_ == nullptr) {
      return ::mediapipe::Status(
          ::mediapipe::StatusCode::kInvalidArgument,
          "SecureRandom is not available.  With \"jitter\" specified, "
          "PacketResamplerCalculator processing cannot proceed.");
    }
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketResamplerCalculator::Process(CalculatorContext* cc) {
  if (cc->InputTimestamp() == Timestamp::PreStream() &&
      cc->Inputs().UsesTags() && cc->Inputs().HasTag("VIDEO_HEADER") &&
      !cc->Inputs().Tag("VIDEO_HEADER").IsEmpty()) {
    video_header_ = cc->Inputs().Tag("VIDEO_HEADER").Get<VideoHeader>();
    video_header_.frame_rate = frame_rate_;
    if (cc->Inputs().Get(input_data_id_).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
  }
  if (jitter_ != 0.0 && random_ != nullptr) {
    MP_RETURN_IF_ERROR(ProcessWithJitter(cc));
  } else {
    MP_RETURN_IF_ERROR(ProcessWithoutJitter(cc));
  }
  last_packet_ = cc->Inputs().Get(input_data_id_).Value();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketResamplerCalculator::ProcessWithJitter(
    CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());
  RET_CHECK_NE(jitter_, 0.0);

  if (first_timestamp_ == Timestamp::Unset()) {
    first_timestamp_ = cc->InputTimestamp();
    next_output_timestamp_ =
        first_timestamp_ + frame_time_usec_ * random_->RandFloat();
    return ::mediapipe::OkStatus();
  }

  LOG_IF(WARNING, frame_time_usec_ <
                      (cc->InputTimestamp() - last_packet_.Timestamp()).Value())
      << "Adding jitter is meaningless when upsampling.";

  const int64 curr_diff =
      (next_output_timestamp_ - cc->InputTimestamp()).Value();
  const int64 last_diff =
      (next_output_timestamp_ - last_packet_.Timestamp()).Value();
  if (curr_diff * last_diff > 0) {
    return ::mediapipe::OkStatus();
  }
  OutputWithinLimits(cc, (std::abs(curr_diff) > std::abs(last_diff)
                              ? last_packet_
                              : cc->Inputs().Get(input_data_id_).Value())
                             .At(next_output_timestamp_));
  next_output_timestamp_ +=
      frame_time_usec_ *
      ((1.0 - jitter_) + 2.0 * jitter_ * random_->RandFloat());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketResamplerCalculator::ProcessWithoutJitter(
    CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());
  RET_CHECK_EQ(jitter_, 0.0);

  if (first_timestamp_ == Timestamp::Unset()) {
    // This is the first packet, initialize the first_timestamp_.
    if (base_timestamp_ == Timestamp::Unset()) {
      // Initialize first_timestamp_ with exactly the first packet timestamp.
      first_timestamp_ = cc->InputTimestamp();
    } else {
      // Initialize first_timestamp_ with the first packet timestamp
      // aligned to the base_timestamp_.
      int64 first_index = MathUtil::SafeRound<int64, double>(
          (cc->InputTimestamp() - base_timestamp_).Seconds() * frame_rate_);
      first_timestamp_ =
          base_timestamp_ + TimestampDiffFromSeconds(first_index / frame_rate_);
    }
    if (cc->Outputs().UsesTags() && cc->Outputs().HasTag("VIDEO_HEADER")) {
      cc->Outputs()
          .Tag("VIDEO_HEADER")
          .Add(new VideoHeader(video_header_), Timestamp::PreStream());
    }
  }
  const Timestamp received_timestamp = cc->InputTimestamp();
  const int64 received_timestamp_idx =
      TimestampToPeriodIndex(received_timestamp);
  // Only consider the received packet if it belongs to the current period
  // (== period_count_) or to a newer one (> period_count_).
  if (received_timestamp_idx >= period_count_) {
    // Fill the empty periods until we are in the same index as the received
    // packet.
    while (received_timestamp_idx > period_count_) {
      OutputWithinLimits(
          cc, last_packet_.At(PeriodIndexToTimestamp(period_count_)));
      ++period_count_;
    }
    // Now, if the received packet has a timestamp larger than the middle of
    // the current period, we can send a packet without waiting. We send the
    // one closer to the middle.
    Timestamp target_timestamp = PeriodIndexToTimestamp(period_count_);
    if (received_timestamp >= target_timestamp) {
      bool have_last_packet = (last_packet_.Timestamp() != Timestamp::Unset());
      bool send_current =
          !have_last_packet || (received_timestamp - target_timestamp <=
                                target_timestamp - last_packet_.Timestamp());
      if (send_current) {
        OutputWithinLimits(
            cc, cc->Inputs().Get(input_data_id_).Value().At(target_timestamp));
      } else {
        OutputWithinLimits(cc, last_packet_.At(target_timestamp));
      }
      ++period_count_;
    }
    // TODO: Add a mechanism to the framework to allow these packets
    // to be output earlier (without waiting for a much later packet to
    // arrive)

    // Update the bound for the next packet.
    cc->Outputs()
        .Get(output_data_id_)
        .SetNextTimestampBound(PeriodIndexToTimestamp(period_count_));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketResamplerCalculator::Close(CalculatorContext* cc) {
  if (!cc->GraphStatus().ok()) {
    return ::mediapipe::OkStatus();
  }
  // Emit the last packet received if we have at least one packet, but
  // haven't sent anything for its period.
  if (first_timestamp_ != Timestamp::Unset() && flush_last_packet_ &&
      TimestampToPeriodIndex(last_packet_.Timestamp()) == period_count_) {
    OutputWithinLimits(cc,
                       last_packet_.At(PeriodIndexToTimestamp(period_count_)));
  }
  return ::mediapipe::OkStatus();
}

Timestamp PacketResamplerCalculator::PeriodIndexToTimestamp(int64 index) const {
  CHECK_EQ(jitter_, 0.0);
  CHECK_NE(first_timestamp_, Timestamp::Unset());
  return first_timestamp_ + TimestampDiffFromSeconds(index / frame_rate_);
}

int64 PacketResamplerCalculator::TimestampToPeriodIndex(
    Timestamp timestamp) const {
  CHECK_EQ(jitter_, 0.0);
  CHECK_NE(first_timestamp_, Timestamp::Unset());
  return MathUtil::SafeRound<int64, double>(
      (timestamp - first_timestamp_).Seconds() * frame_rate_);
}

void PacketResamplerCalculator::OutputWithinLimits(CalculatorContext* cc,
                                                   const Packet& packet) const {
  TimestampDiff margin((round_limits_) ? frame_time_usec_ / 2 : 0);
  if (packet.Timestamp() >= start_time_ - margin &&
      packet.Timestamp() < end_time_ + margin) {
    cc->Outputs().Get(output_data_id_).AddPacket(packet);
  }
}

}  // namespace mediapipe
