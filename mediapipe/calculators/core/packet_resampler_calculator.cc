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

#include "mediapipe/calculators/core/packet_resampler_calculator.h"

#include <memory>

namespace {
// Reflect an integer against the lower and upper bound of an interval.
int64 ReflectBetween(int64 ts, int64 ts_min, int64 ts_max) {
  if (ts < ts_min) return 2 * ts_min - ts - 1;
  if (ts >= ts_max) return 2 * ts_max - ts - 1;
  return ts;
}

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
  jitter_with_reflection_ = resampler_options.jitter_with_reflection();

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
  jitter_usec_ = static_cast<int64>(1000000.0 * jitter_ / frame_rate_);
  RET_CHECK_LE(jitter_usec_, frame_time_usec_);

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
    packet_reservoir_random_ = CreateSecureRandom(seed);
  }
  packet_reservoir_ =
      std::make_unique<PacketReservoir>(packet_reservoir_random_.get());
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
    // Packet reservior is used to make sure there's an output for every period,
    // e.g. partial period at the end of the stream.
    if (packet_reservoir_->IsEnabled() &&
        (first_timestamp_ == Timestamp::Unset() ||
         (cc->InputTimestamp() - next_output_timestamp_min_).Value() >= 0)) {
      auto curr_packet = cc->Inputs().Get(input_data_id_).Value();
      packet_reservoir_->AddSample(curr_packet);
    }
    MP_RETURN_IF_ERROR(ProcessWithJitter(cc));
  } else {
    MP_RETURN_IF_ERROR(ProcessWithoutJitter(cc));
  }
  last_packet_ = cc->Inputs().Get(input_data_id_).Value();
  return ::mediapipe::OkStatus();
}

void PacketResamplerCalculator::InitializeNextOutputTimestampWithJitter() {
  next_output_timestamp_min_ = first_timestamp_;
  if (jitter_with_reflection_) {
    next_output_timestamp_ =
        first_timestamp_ + random_->UnbiasedUniform64(frame_time_usec_);
    return;
  }
  next_output_timestamp_ =
      first_timestamp_ + frame_time_usec_ * random_->RandFloat();
}

void PacketResamplerCalculator::UpdateNextOutputTimestampWithJitter() {
  packet_reservoir_->Clear();
  if (jitter_with_reflection_) {
    next_output_timestamp_min_ += frame_time_usec_;
    Timestamp next_output_timestamp_max_ =
        next_output_timestamp_min_ + frame_time_usec_;

    next_output_timestamp_ += frame_time_usec_ +
                              random_->UnbiasedUniform64(2 * jitter_usec_ + 1) -
                              jitter_usec_;
    next_output_timestamp_ = Timestamp(ReflectBetween(
        next_output_timestamp_.Value(), next_output_timestamp_min_.Value(),
        next_output_timestamp_max_.Value()));
    CHECK_GE(next_output_timestamp_, next_output_timestamp_min_);
    CHECK_LT(next_output_timestamp_, next_output_timestamp_max_);
    return;
  }
  packet_reservoir_->Disable();
  next_output_timestamp_ +=
      frame_time_usec_ *
      ((1.0 - jitter_) + 2.0 * jitter_ * random_->RandFloat());
}

::mediapipe::Status PacketResamplerCalculator::ProcessWithJitter(
    CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());
  RET_CHECK_NE(jitter_, 0.0);

  if (first_timestamp_ == Timestamp::Unset()) {
    first_timestamp_ = cc->InputTimestamp();
    InitializeNextOutputTimestampWithJitter();
    if (first_timestamp_ == next_output_timestamp_) {
      OutputWithinLimits(
          cc,
          cc->Inputs().Get(input_data_id_).Value().At(next_output_timestamp_));
      UpdateNextOutputTimestampWithJitter();
    }
    return ::mediapipe::OkStatus();
  }

  if (frame_time_usec_ <
      (cc->InputTimestamp() - last_packet_.Timestamp()).Value()) {
    LOG_FIRST_N(WARNING, 2)
        << "Adding jitter is not very useful when upsampling.";
  }

  while (true) {
    const int64 last_diff =
        (next_output_timestamp_ - last_packet_.Timestamp()).Value();
    RET_CHECK_GT(last_diff, 0);
    const int64 curr_diff =
        (next_output_timestamp_ - cc->InputTimestamp()).Value();
    if (curr_diff > 0) {
      break;
    }
    OutputWithinLimits(cc, (std::abs(curr_diff) > last_diff
                                ? last_packet_
                                : cc->Inputs().Get(input_data_id_).Value())
                               .At(next_output_timestamp_));
    UpdateNextOutputTimestampWithJitter();
  }
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
  if (!packet_reservoir_->IsEmpty()) {
    OutputWithinLimits(cc, packet_reservoir_->GetSample());
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
