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

#include <algorithm>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/ret_check.h"

namespace {
// Reflect an integer against the lower and upper bound of an interval.
int64_t ReflectBetween(int64_t ts, int64_t ts_min, int64_t ts_max) {
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

constexpr char kSeedTag[] = "SEED";
constexpr char kVideoHeaderTag[] = "VIDEO_HEADER";
constexpr char kOptionsTag[] = "OPTIONS";

// Returns a TimestampDiff (assuming microseconds) corresponding to the
// given time in seconds.
TimestampDiff TimestampDiffFromSeconds(double seconds) {
  return TimestampDiff(MathUtil::SafeRound<int64_t, double>(
      seconds * Timestamp::kTimestampUnitsPerSecond));
}
}  // namespace

absl::Status PacketResamplerCalculator::GetContract(CalculatorContract* cc) {
  const auto& resampler_options =
      cc->Options<PacketResamplerCalculatorOptions>();
  if (cc->InputSidePackets().HasTag(kOptionsTag)) {
    cc->InputSidePackets().Tag(kOptionsTag).Set<CalculatorOptions>();
  }
  CollectionItemId input_data_id = cc->Inputs().GetId("DATA", 0);
  if (!input_data_id.IsValid()) {
    input_data_id = cc->Inputs().GetId("", 0);
  }
  cc->Inputs().Get(input_data_id).SetAny();
  if (cc->Inputs().HasTag(kVideoHeaderTag)) {
    cc->Inputs().Tag(kVideoHeaderTag).Set<VideoHeader>();
  }

  CollectionItemId output_data_id = cc->Outputs().GetId("DATA", 0);
  if (!output_data_id.IsValid()) {
    output_data_id = cc->Outputs().GetId("", 0);
  }
  cc->Outputs().Get(output_data_id).SetSameAs(&cc->Inputs().Get(input_data_id));
  if (cc->Outputs().HasTag(kVideoHeaderTag)) {
    RET_CHECK(resampler_options.max_frame_rate() <= 0)
        << "VideoHeader output is not supported with max_frame_rate.";
    cc->Outputs().Tag(kVideoHeaderTag).Set<VideoHeader>();
  }

  if (resampler_options.jitter() != 0.0) {
    RET_CHECK_GT(resampler_options.jitter(), 0.0);
    RET_CHECK_LE(resampler_options.jitter(), 1.0);
    RET_CHECK(cc->InputSidePackets().HasTag(kSeedTag));
    cc->InputSidePackets().Tag(kSeedTag).Set<std::string>();
  }
  return absl::OkStatus();
}

absl::Status PacketResamplerCalculator::UpdateFrameRate(
    const PacketResamplerCalculatorOptions& resampler_options,
    double frame_rate) {
  frame_rate_ = frame_rate;
  if (resampler_options.max_frame_rate() > 0) {
    frame_rate_ = std::min(frame_rate_, resampler_options.max_frame_rate());
  }
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

  frame_time_usec_ = static_cast<int64_t>(1000000.0 / frame_rate_);
  jitter_usec_ = static_cast<int64_t>(1000000.0 * jitter_ / frame_rate_);
  RET_CHECK_LE(jitter_usec_, frame_time_usec_);

  video_header_.frame_rate = frame_rate_;
  return absl::OkStatus();
}

absl::Status PacketResamplerCalculator::Open(CalculatorContext* cc) {
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

  RET_CHECK_OK(
      UpdateFrameRate(resampler_options, resampler_options.frame_rate()));

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

  strategy_ = GetSamplingStrategy(resampler_options);

  return strategy_->Open(cc);
}

absl::Status PacketResamplerCalculator::Process(CalculatorContext* cc) {
  const auto resampler_options =
      tool::RetrieveOptions(cc->Options<PacketResamplerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");

  if (cc->InputTimestamp() == Timestamp::PreStream() &&
      cc->Inputs().UsesTags() && cc->Inputs().HasTag(kVideoHeaderTag) &&
      !cc->Inputs().Tag(kVideoHeaderTag).IsEmpty()) {
    video_header_ = cc->Inputs().Tag(kVideoHeaderTag).Get<VideoHeader>();
    if (resampler_options.use_input_frame_rate()) {
      RET_CHECK_OK(
          UpdateFrameRate(resampler_options, video_header_.frame_rate));
    }
    video_header_.frame_rate = frame_rate_;
    if (cc->Inputs().Get(input_data_id_).IsEmpty()) {
      return absl::OkStatus();
    }
  }
  if (!header_sent_ && cc->Outputs().UsesTags() &&
      cc->Outputs().HasTag(kVideoHeaderTag)) {
    cc->Outputs()
        .Tag(kVideoHeaderTag)
        .Add(new VideoHeader(video_header_), Timestamp::PreStream());
    header_sent_ = true;
  }

  MP_RETURN_IF_ERROR(strategy_->Process(cc));

  last_packet_ = cc->Inputs().Get(input_data_id_).Value();

  return absl::OkStatus();
}

absl::Status PacketResamplerCalculator::Close(CalculatorContext* cc) {
  if (!cc->GraphStatus().ok()) {
    return absl::OkStatus();
  }

  return strategy_->Close(cc);
}

std::unique_ptr<PacketResamplerStrategy>
PacketResamplerCalculator::GetSamplingStrategy(
    const PacketResamplerCalculatorOptions& options) {
  if (options.reproducible_sampling()) {
    if (!options.jitter_with_reflection()) {
      ABSL_LOG(WARNING)
          << "reproducible_sampling enabled w/ jitter_with_reflection "
             "disabled. "
          << "reproducible_sampling always uses jitter with reflection, "
          << "Ignoring jitter_with_reflection setting.";
    }
    return absl::make_unique<ReproducibleJitterWithReflectionStrategy>(this);
  }

  if (options.jitter() == 0) {
    return absl::make_unique<NoJitterStrategy>(this);
  }

  if (options.jitter_with_reflection()) {
    return absl::make_unique<LegacyJitterWithReflectionStrategy>(this);
  }

  // With jitter and no reflection.
  return absl::make_unique<JitterWithoutReflectionStrategy>(this);
}

Timestamp PacketResamplerCalculator::PeriodIndexToTimestamp(
    int64_t index) const {
  ABSL_CHECK_EQ(jitter_, 0.0);
  ABSL_CHECK_NE(first_timestamp_, Timestamp::Unset());
  return first_timestamp_ + TimestampDiffFromSeconds(index / frame_rate_);
}

int64_t PacketResamplerCalculator::TimestampToPeriodIndex(
    Timestamp timestamp) const {
  ABSL_CHECK_EQ(jitter_, 0.0);
  ABSL_CHECK_NE(first_timestamp_, Timestamp::Unset());
  return MathUtil::SafeRound<int64_t, double>(
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

absl::Status LegacyJitterWithReflectionStrategy::Open(CalculatorContext* cc) {
  const auto resampler_options =
      tool::RetrieveOptions(cc->Options<PacketResamplerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");

  if (resampler_options.output_header() !=
      PacketResamplerCalculatorOptions::NONE) {
    ABSL_LOG(WARNING)
        << "VideoHeader::frame_rate holds the target value and not "
           "the actual value.";
  }

  if (calculator_->flush_last_packet_) {
    ABSL_LOG(WARNING)
        << "PacketResamplerCalculatorOptions.flush_last_packet is "
           "ignored, because we are adding jitter.";
  }

  const auto& seed = cc->InputSidePackets().Tag(kSeedTag).Get<std::string>();
  random_ = CreateSecureRandom(seed);
  if (random_ == nullptr) {
    return absl::InvalidArgumentError(
        "SecureRandom is not available.  With \"jitter\" specified, "
        "PacketResamplerCalculator processing cannot proceed.");
  }

  packet_reservoir_random_ = CreateSecureRandom(seed);
  packet_reservoir_ =
      std::make_unique<PacketReservoir>(packet_reservoir_random_.get());

  return absl::OkStatus();
}
absl::Status LegacyJitterWithReflectionStrategy::Close(CalculatorContext* cc) {
  if (!packet_reservoir_->IsEmpty()) {
    ABSL_LOG(INFO) << "Emitting pack from reservoir.";
    calculator_->OutputWithinLimits(cc, packet_reservoir_->GetSample());
  }
  return absl::OkStatus();
}
absl::Status LegacyJitterWithReflectionStrategy::Process(
    CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());

  if (packet_reservoir_->IsEnabled() &&
      (first_timestamp_ == Timestamp::Unset() ||
       (cc->InputTimestamp() - next_output_timestamp_min_).Value() >= 0)) {
    auto curr_packet = cc->Inputs().Get(calculator_->input_data_id_).Value();
    packet_reservoir_->AddSample(curr_packet);
  }

  if (first_timestamp_ == Timestamp::Unset()) {
    first_timestamp_ = cc->InputTimestamp();
    InitializeNextOutputTimestampWithJitter();
    if (first_timestamp_ == next_output_timestamp_) {
      calculator_->OutputWithinLimits(cc, cc->Inputs()
                                              .Get(calculator_->input_data_id_)
                                              .Value()
                                              .At(next_output_timestamp_));
      UpdateNextOutputTimestampWithJitter();
    }
    return absl::OkStatus();
  }

  if (calculator_->frame_time_usec_ <
      (cc->InputTimestamp() - calculator_->last_packet_.Timestamp()).Value()) {
    ABSL_LOG_FIRST_N(WARNING, 2)
        << "Adding jitter is not very useful when upsampling.";
  }

  while (true) {
    const int64_t last_diff =
        (next_output_timestamp_ - calculator_->last_packet_.Timestamp())
            .Value();
    RET_CHECK_GT(last_diff, 0);
    const int64_t curr_diff =
        (next_output_timestamp_ - cc->InputTimestamp()).Value();
    if (curr_diff > 0) {
      break;
    }
    calculator_->OutputWithinLimits(
        cc, (std::abs(curr_diff) > last_diff
                 ? calculator_->last_packet_
                 : cc->Inputs().Get(calculator_->input_data_id_).Value())
                .At(next_output_timestamp_));
    UpdateNextOutputTimestampWithJitter();
    // From now on every time a packet is emitted the timestamp of the next
    // packet becomes known; that timestamp is stored in next_output_timestamp_.
    // The only exception to this rule is the packet emitted from Close() which
    // can only happen when jitter_with_reflection is enabled but in this case
    // next_output_timestamp_min_ is a non-decreasing lower bound of any
    // subsequent packet.
    const Timestamp timestamp_bound = next_output_timestamp_min_;
    cc->Outputs()
        .Get(calculator_->output_data_id_)
        .SetNextTimestampBound(timestamp_bound);
  }
  return absl::OkStatus();
}

void LegacyJitterWithReflectionStrategy::
    InitializeNextOutputTimestampWithJitter() {
  next_output_timestamp_min_ = first_timestamp_;
  next_output_timestamp_ =
      first_timestamp_ +
      random_->UnbiasedUniform64(calculator_->frame_time_usec_);
}

void LegacyJitterWithReflectionStrategy::UpdateNextOutputTimestampWithJitter() {
  packet_reservoir_->Clear();
  next_output_timestamp_min_ += calculator_->frame_time_usec_;
  Timestamp next_output_timestamp_max_ =
      next_output_timestamp_min_ + calculator_->frame_time_usec_;

  next_output_timestamp_ +=
      calculator_->frame_time_usec_ +
      random_->UnbiasedUniform64(2 * calculator_->jitter_usec_ + 1) -
      calculator_->jitter_usec_;
  next_output_timestamp_ = Timestamp(ReflectBetween(
      next_output_timestamp_.Value(), next_output_timestamp_min_.Value(),
      next_output_timestamp_max_.Value()));
  ABSL_CHECK_GE(next_output_timestamp_, next_output_timestamp_min_);
  ABSL_CHECK_LT(next_output_timestamp_, next_output_timestamp_max_);
}

absl::Status ReproducibleJitterWithReflectionStrategy::Open(
    CalculatorContext* cc) {
  const auto resampler_options =
      tool::RetrieveOptions(cc->Options<PacketResamplerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");

  if (resampler_options.output_header() !=
      PacketResamplerCalculatorOptions::NONE) {
    ABSL_LOG(WARNING)
        << "VideoHeader::frame_rate holds the target value and not "
           "the actual value.";
  }

  if (calculator_->flush_last_packet_) {
    ABSL_LOG(WARNING)
        << "PacketResamplerCalculatorOptions.flush_last_packet is "
           "ignored, because we are adding jitter.";
  }

  const auto& seed = cc->InputSidePackets().Tag(kSeedTag).Get<std::string>();
  random_ = CreateSecureRandom(seed);
  if (random_ == nullptr) {
    return absl::InvalidArgumentError(
        "SecureRandom is not available.  With \"jitter\" specified, "
        "PacketResamplerCalculator processing cannot proceed.");
  }

  return absl::OkStatus();
}
absl::Status ReproducibleJitterWithReflectionStrategy::Close(
    CalculatorContext* cc) {
  // If last packet is non-empty and a packet hasn't been emitted for this
  // period, emit the last packet.
  if (!calculator_->last_packet_.IsEmpty() && !packet_emitted_this_period_) {
    calculator_->OutputWithinLimits(
        cc, calculator_->last_packet_.At(next_output_timestamp_));
  }
  return absl::OkStatus();
}
absl::Status ReproducibleJitterWithReflectionStrategy::Process(
    CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());

  Packet current_packet = cc->Inputs().Get(calculator_->input_data_id_).Value();

  if (calculator_->last_packet_.IsEmpty()) {
    // last_packet is empty, this is the first packet of the stream.

    InitializeNextOutputTimestamp(current_packet.Timestamp());

    // If next_output_timestamp_ happens to fall before current_packet, emit
    // current packet.  Only a single packet can be emitted at the beginning
    // of the stream.
    if (next_output_timestamp_ < current_packet.Timestamp()) {
      calculator_->OutputWithinLimits(
          cc, current_packet.At(next_output_timestamp_));
      packet_emitted_this_period_ = true;
    }

    return absl::OkStatus();
  }

  // Last packet is set, so we are mid-stream.
  if (calculator_->frame_time_usec_ <
      (current_packet.Timestamp() - calculator_->last_packet_.Timestamp())
          .Value()) {
    // Note, if the stream is upsampling, this could lead to the same packet
    // being emitted twice.  Upsampling and jitter doesn't make much sense
    // but does technically work.
    ABSL_LOG_FIRST_N(WARNING, 2)
        << "Adding jitter is not very useful when upsampling.";
  }

  // Since we may be upsampling, we need to iteratively advance the
  // next_output_timestamp_ one period at a time until it reaches the period
  // current_packet is in.  During this process, last_packet and/or
  // current_packet may be repeatly emitted.

  UpdateNextOutputTimestamp(current_packet.Timestamp());

  while (!packet_emitted_this_period_ &&
         next_output_timestamp_ <= current_packet.Timestamp()) {
    // last_packet < next_output_timestamp_ <= current_packet,
    // so emit the closest packet.
    Packet packet_to_emit =
        current_packet.Timestamp() - next_output_timestamp_ <
                next_output_timestamp_ - calculator_->last_packet_.Timestamp()
            ? current_packet
            : calculator_->last_packet_;
    calculator_->OutputWithinLimits(cc,
                                    packet_to_emit.At(next_output_timestamp_));

    packet_emitted_this_period_ = true;

    // If we are upsampling, packet_emitted_this_period_ can be reset by
    // the following UpdateNext and the loop will iterate.
    UpdateNextOutputTimestamp(current_packet.Timestamp());
  }

  // Set the bounds on the output stream.  Note, if we emitted a packet
  // above, it will already be set at next_output_timestamp_ + 1, in which
  // case we have to skip setting it.
  if (cc->Outputs().Get(calculator_->output_data_id_).NextTimestampBound() <
      next_output_timestamp_) {
    cc->Outputs()
        .Get(calculator_->output_data_id_)
        .SetNextTimestampBound(next_output_timestamp_);
  }
  return absl::OkStatus();
}

void ReproducibleJitterWithReflectionStrategy::InitializeNextOutputTimestamp(
    Timestamp current_timestamp) {
  if (next_output_timestamp_min_ != Timestamp::Unset()) {
    return;
  }

  next_output_timestamp_min_ = Timestamp(0);
  next_output_timestamp_ =
      Timestamp(GetNextRandom(calculator_->frame_time_usec_));

  // While the current timestamp is ahead of the max (i.e. min + frame_time),
  // fast-forward.
  while (current_timestamp >=
         next_output_timestamp_min_ + calculator_->frame_time_usec_) {
    packet_emitted_this_period_ = true;  // Force update...
    UpdateNextOutputTimestamp(current_timestamp);
  }
}

void ReproducibleJitterWithReflectionStrategy::UpdateNextOutputTimestamp(
    Timestamp current_timestamp) {
  if (packet_emitted_this_period_ &&
      current_timestamp >=
          next_output_timestamp_min_ + calculator_->frame_time_usec_) {
    next_output_timestamp_min_ += calculator_->frame_time_usec_;
    Timestamp next_output_timestamp_max_ =
        next_output_timestamp_min_ + calculator_->frame_time_usec_;

    next_output_timestamp_ += calculator_->frame_time_usec_ +
                              GetNextRandom(2 * calculator_->jitter_usec_ + 1) -
                              calculator_->jitter_usec_;
    next_output_timestamp_ = Timestamp(ReflectBetween(
        next_output_timestamp_.Value(), next_output_timestamp_min_.Value(),
        next_output_timestamp_max_.Value()));

    packet_emitted_this_period_ = false;
  }
}

absl::Status JitterWithoutReflectionStrategy::Open(CalculatorContext* cc) {
  const auto resampler_options =
      tool::RetrieveOptions(cc->Options<PacketResamplerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");

  if (resampler_options.output_header() !=
      PacketResamplerCalculatorOptions::NONE) {
    ABSL_LOG(WARNING)
        << "VideoHeader::frame_rate holds the target value and not "
           "the actual value.";
  }

  if (calculator_->flush_last_packet_) {
    ABSL_LOG(WARNING)
        << "PacketResamplerCalculatorOptions.flush_last_packet is "
           "ignored, because we are adding jitter.";
  }

  const auto& seed = cc->InputSidePackets().Tag(kSeedTag).Get<std::string>();
  random_ = CreateSecureRandom(seed);
  if (random_ == nullptr) {
    return absl::InvalidArgumentError(
        "SecureRandom is not available.  With \"jitter\" specified, "
        "PacketResamplerCalculator processing cannot proceed.");
  }

  packet_reservoir_random_ = CreateSecureRandom(seed);
  packet_reservoir_ =
      absl::make_unique<PacketReservoir>(packet_reservoir_random_.get());

  return absl::OkStatus();
}
absl::Status JitterWithoutReflectionStrategy::Close(CalculatorContext* cc) {
  if (!packet_reservoir_->IsEmpty()) {
    calculator_->OutputWithinLimits(cc, packet_reservoir_->GetSample());
  }
  return absl::OkStatus();
}
absl::Status JitterWithoutReflectionStrategy::Process(CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());

  // Packet reservior is used to make sure there's an output for every period,
  // e.g. partial period at the end of the stream.
  if (packet_reservoir_->IsEnabled() &&
      (calculator_->first_timestamp_ == Timestamp::Unset() ||
       (cc->InputTimestamp() - next_output_timestamp_min_).Value() >= 0)) {
    auto curr_packet = cc->Inputs().Get(calculator_->input_data_id_).Value();
    packet_reservoir_->AddSample(curr_packet);
  }

  if (calculator_->first_timestamp_ == Timestamp::Unset()) {
    calculator_->first_timestamp_ = cc->InputTimestamp();
    InitializeNextOutputTimestamp();
    if (calculator_->first_timestamp_ == next_output_timestamp_) {
      calculator_->OutputWithinLimits(cc, cc->Inputs()
                                              .Get(calculator_->input_data_id_)
                                              .Value()
                                              .At(next_output_timestamp_));
      UpdateNextOutputTimestamp();
    }
    return absl::OkStatus();
  }

  if (calculator_->frame_time_usec_ <
      (cc->InputTimestamp() - calculator_->last_packet_.Timestamp()).Value()) {
    ABSL_LOG_FIRST_N(WARNING, 2)
        << "Adding jitter is not very useful when upsampling.";
  }

  while (true) {
    const int64_t last_diff =
        (next_output_timestamp_ - calculator_->last_packet_.Timestamp())
            .Value();
    RET_CHECK_GT(last_diff, 0);
    const int64_t curr_diff =
        (next_output_timestamp_ - cc->InputTimestamp()).Value();
    if (curr_diff > 0) {
      break;
    }
    calculator_->OutputWithinLimits(
        cc, (std::abs(curr_diff) > last_diff
                 ? calculator_->last_packet_
                 : cc->Inputs().Get(calculator_->input_data_id_).Value())
                .At(next_output_timestamp_));
    UpdateNextOutputTimestamp();
    cc->Outputs()
        .Get(calculator_->output_data_id_)
        .SetNextTimestampBound(next_output_timestamp_);
  }
  return absl::OkStatus();
}

void JitterWithoutReflectionStrategy::InitializeNextOutputTimestamp() {
  next_output_timestamp_min_ = calculator_->first_timestamp_;
  next_output_timestamp_ = calculator_->first_timestamp_ +
                           calculator_->frame_time_usec_ * random_->RandFloat();
}

void JitterWithoutReflectionStrategy::UpdateNextOutputTimestamp() {
  packet_reservoir_->Clear();
  packet_reservoir_->Disable();
  next_output_timestamp_ += calculator_->frame_time_usec_ *
                            ((1.0 - calculator_->jitter_) +
                             2.0 * calculator_->jitter_ * random_->RandFloat());
}

absl::Status NoJitterStrategy::Open(CalculatorContext* cc) {
  const auto resampler_options =
      tool::RetrieveOptions(cc->Options<PacketResamplerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");
  base_timestamp_ = resampler_options.has_base_timestamp()
                        ? Timestamp(resampler_options.base_timestamp())
                        : Timestamp::Unset();

  period_count_ = 0;

  return absl::OkStatus();
}
absl::Status NoJitterStrategy::Close(CalculatorContext* cc) {
  // Emit the last packet received if we have at least one packet, but
  // haven't sent anything for its period.
  if (calculator_->first_timestamp_ != Timestamp::Unset() &&
      calculator_->flush_last_packet_ &&
      calculator_->TimestampToPeriodIndex(
          calculator_->last_packet_.Timestamp()) == period_count_) {
    calculator_->OutputWithinLimits(
        cc, calculator_->last_packet_.At(
                calculator_->PeriodIndexToTimestamp(period_count_)));
  }
  return absl::OkStatus();
}
absl::Status NoJitterStrategy::Process(CalculatorContext* cc) {
  RET_CHECK_GT(cc->InputTimestamp(), Timestamp::PreStream());

  if (calculator_->first_timestamp_ == Timestamp::Unset()) {
    // This is the first packet, initialize the first_timestamp_.
    if (base_timestamp_ == Timestamp::Unset()) {
      // Initialize first_timestamp_ with exactly the first packet timestamp.
      calculator_->first_timestamp_ = cc->InputTimestamp();
    } else {
      // Initialize first_timestamp_ with the first packet timestamp
      // aligned to the base_timestamp_.
      int64_t first_index = MathUtil::SafeRound<int64_t, double>(
          (cc->InputTimestamp() - base_timestamp_).Seconds() *
          calculator_->frame_rate_);
      calculator_->first_timestamp_ =
          base_timestamp_ +
          TimestampDiffFromSeconds(first_index / calculator_->frame_rate_);
    }
  }
  const Timestamp received_timestamp = cc->InputTimestamp();
  const int64_t received_timestamp_idx =
      calculator_->TimestampToPeriodIndex(received_timestamp);
  // Only consider the received packet if it belongs to the current period
  // (== period_count_) or to a newer one (> period_count_).
  if (received_timestamp_idx >= period_count_) {
    // Fill the empty periods until we are in the same index as the received
    // packet.
    while (received_timestamp_idx > period_count_) {
      calculator_->OutputWithinLimits(
          cc, calculator_->last_packet_.At(
                  calculator_->PeriodIndexToTimestamp(period_count_)));
      ++period_count_;
    }
    // Now, if the received packet has a timestamp larger than the middle of
    // the current period, we can send a packet without waiting. We send the
    // one closer to the middle.
    Timestamp target_timestamp =
        calculator_->PeriodIndexToTimestamp(period_count_);
    if (received_timestamp >= target_timestamp) {
      bool have_last_packet =
          (calculator_->last_packet_.Timestamp() != Timestamp::Unset());
      bool send_current =
          !have_last_packet ||
          (received_timestamp - target_timestamp <=
           target_timestamp - calculator_->last_packet_.Timestamp());
      if (send_current) {
        calculator_->OutputWithinLimits(cc,
                                        cc->Inputs()
                                            .Get(calculator_->input_data_id_)
                                            .Value()
                                            .At(target_timestamp));
      } else {
        calculator_->OutputWithinLimits(
            cc, calculator_->last_packet_.At(target_timestamp));
      }
      ++period_count_;
    }
    // TODO: Add a mechanism to the framework to allow these packets
    // to be output earlier (without waiting for a much later packet to
    // arrive)

    // Update the bound for the next packet.
    cc->Outputs()
        .Get(calculator_->output_data_id_)
        .SetNextTimestampBound(
            calculator_->PeriodIndexToTimestamp(period_count_));
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
