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
//
// Defines TimeSeriesFramerCalculator.
#include <math.h>

#include <vector>

#include "Eigen/Core"
#include "absl/log/absl_check.h"
#include "audio/dsp/window_functions.h"
#include "mediapipe/calculators/audio/time_series_framer_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

// MediaPipe Calculator for framing a (vector-valued) input time series,
// i.e. for breaking an input time series into fixed-size, possibly
// overlapping, frames.  The output stream's frame duration is
// specified by frame_duration_seconds in the
// TimeSeriesFramerCalculatorOptions, and the output's overlap is
// specified by frame_overlap_seconds.
//
// This calculator assumes that the input timestamps refer to the
// first sample in each Matrix.  The output timestamps follow this
// same convention.
//
// All output frames will have exactly the same number of samples: the number of
// samples that approximates frame_duration_seconds most closely.
//
// Similarly, frame overlap is by default the (fixed) number of samples
// approximating frame_overlap_seconds most closely.  But if
// emulate_fractional_frame_overlap is set to true, frame overlap is a variable
// number of samples instead, such that the long-term average step between
// frames is the difference between the (nominal) frame_duration_seconds and
// frame_overlap_seconds.
//
// If pad_final_packet is true, all input samples will be emitted and the final
// packet will be zero padded as necessary.  If pad_final_packet is false, some
// samples may be dropped at the end of the stream.
//
// If use_local_timestamp is true, the output packet's timestamp is based on the
// last sample of the packet. The timestamp of this sample is inferred by
// input_packet_timesamp + local_sample_index / sampling_rate_. If false, the
// output packet's timestamp is based on the cumulative timestamping, which is
// done by adopting the timestamp of the first sample of the packet and this
// sample's timestamp is inferred by initial_input_timestamp_ +
// cumulative_completed_samples / sample_rate_.
class TimeSeriesFramerCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Matrix>(
        // Input stream with TimeSeriesHeader.
    );
    cc->Outputs().Index(0).Set<Matrix>(
        // Fixed length time series Packets with TimeSeriesHeader.
    );
    return absl::OkStatus();
  }

  // Returns FAIL if the input stream header is invalid.
  absl::Status Open(CalculatorContext* cc) override;

  // Outputs as many framed packets as possible given the accumulated
  // input.  Always returns OK.
  absl::Status Process(CalculatorContext* cc) override;

  // Flushes any remaining samples in a zero-padded packet.  Always
  // returns OK.
  absl::Status Close(CalculatorContext* cc) override;

 private:
  Timestamp CurrentOutputTimestamp() {
    if (use_local_timestamp_) {
      return current_timestamp_;
    }
    return CumulativeOutputTimestamp();
  }

  Timestamp CumulativeOutputTimestamp() {
    return initial_input_timestamp_ +
           round(cumulative_completed_samples_ / sample_rate_ *
                 Timestamp::kTimestampUnitsPerSecond);
  }

  // The number of input samples to advance after the current output frame is
  // emitted.
  int next_frame_step_samples() const {
    // All numbers are in input samples.
    const int64_t current_output_frame_start = static_cast<int64_t>(
        round(cumulative_output_frames_ * average_frame_step_samples_));
    ABSL_CHECK_EQ(current_output_frame_start, cumulative_completed_samples_);
    const int64_t next_output_frame_start = static_cast<int64_t>(
        round((cumulative_output_frames_ + 1) * average_frame_step_samples_));
    return next_output_frame_start - current_output_frame_start;
  }

  double sample_rate_;
  bool pad_final_packet_;
  int frame_duration_samples_;
  // The advance, in input samples, between the start of successive output
  // frames. This may be a non-integer average value if
  // emulate_fractional_frame_overlap is true.
  double average_frame_step_samples_;
  int samples_still_to_drop_;
  int64_t cumulative_output_frames_;
  // "Completed" samples are samples that are no longer needed because
  // the framer has completely stepped past them (taking into account
  // any overlap).
  int64_t cumulative_completed_samples_;
  Timestamp initial_input_timestamp_;
  // The current timestamp is updated along with the incoming packets.
  Timestamp current_timestamp_;

  // Samples are buffered in a vector of sample blocks.
  class SampleBlockBuffer {
   public:
    // Initializes the buffer.
    void Init(double sample_rate, int num_channels) {
      ts_units_per_sample_ = Timestamp::kTimestampUnitsPerSecond / sample_rate;
      num_channels_ = num_channels;
      num_samples_ = 0;
      first_block_offset_ = 0;
    }

    // Number of channels, equal to the number of rows in each Matrix.
    int num_channels() const { return num_channels_; }
    // Total number of available samples over all blocks.
    int num_samples() const { return num_samples_; }

    // Pushes a new block of samples on the back of the buffer with `timestamp`
    // being the input timestamp of the packet containing the Matrix.
    void Push(const Matrix& samples, Timestamp timestamp);
    // Copies `count` samples from the front of the buffer. If there are fewer
    // samples than this, the result is zero padded to have `count` samples.
    // The timestamp of the last copied sample is written to *last_timestamp.
    // This output is used below to update `current_timestamp_`, which is only
    // used when `use_local_timestamp` is true.
    Matrix CopySamples(int count, Timestamp* last_timestamp) const;
    // Drops `count` samples from the front of the buffer. If `count` exceeds
    // `num_samples()`, the buffer is emptied.  Returns how many samples were
    // dropped.
    int DropSamples(int count);

   private:
    struct Block {
      // Matrix of num_channels rows by num_samples columns, a block of possibly
      // multiple samples.
      Matrix samples;
      // Timestamp of the first sample in the Block. This comes from the input
      // packet's timestamp that contains this Matrix.
      Timestamp timestamp;

      Block() : timestamp(Timestamp::Unstarted()) {}
      Block(const Matrix& samples, Timestamp timestamp)
          : samples(samples), timestamp(timestamp) {}
      int num_samples() const { return samples.cols(); }
    };
    std::vector<Block> blocks_;
    // Number of timestamp units per sample. Used to compute timestamps as
    // nth sample timestamp = base_timestamp + round(ts_units_per_sample_ * n).
    double ts_units_per_sample_;
    // Number of rows in each Matrix.
    int num_channels_;
    // The total number of samples over all blocks, equal to
    // (sum_i blocks_[i].num_samples()) - first_block_offset_.
    int num_samples_;
    // The number of samples in the first block that have been discarded. This
    // way we can cheaply represent "partially discarding" a block.
    int first_block_offset_;
  } sample_buffer_;

  bool use_window_;
  Eigen::RowVectorXf window_;

  bool use_local_timestamp_;
};
REGISTER_CALCULATOR(TimeSeriesFramerCalculator);

void TimeSeriesFramerCalculator::SampleBlockBuffer::Push(const Matrix& samples,
                                                         Timestamp timestamp) {
  num_samples_ += samples.cols();
  blocks_.emplace_back(samples, timestamp);
}

Matrix TimeSeriesFramerCalculator::SampleBlockBuffer::CopySamples(
    int count, Timestamp* last_timestamp) const {
  Matrix copied(num_channels_, count);

  if (!blocks_.empty()) {
    int num_copied = 0;
    // First block has an offset for samples that have been discarded.
    int offset = first_block_offset_;
    int n;
    Timestamp last_block_ts;
    int last_sample_index;

    for (auto it = blocks_.begin(); it != blocks_.end() && count > 0; ++it) {
      n = std::min(it->num_samples() - offset, count);
      // Copy `n` samples from the next block.
      copied.middleCols(num_copied, n) = it->samples.middleCols(offset, n);
      count -= n;
      num_copied += n;
      last_block_ts = it->timestamp;
      last_sample_index = offset + n - 1;
      offset = 0;  // No samples have been discarded in subsequent blocks.
    }

    // Compute the timestamp of the last copied sample.
    *last_timestamp =
        last_block_ts + std::round(ts_units_per_sample_ * last_sample_index);
  }

  if (count > 0) {
    copied.rightCols(count).setZero();  // Zero pad if needed.
  }

  return copied;
}

int TimeSeriesFramerCalculator::SampleBlockBuffer::DropSamples(int count) {
  if (blocks_.empty()) {
    return 0;
  }

  auto block_it = blocks_.begin();
  if (first_block_offset_ + count < block_it->num_samples()) {
    // `count` is less than the remaining samples in the first block.
    first_block_offset_ += count;
    num_samples_ -= count;
    return count;
  }

  int num_samples_dropped = block_it->num_samples() - first_block_offset_;
  count -= num_samples_dropped;
  first_block_offset_ = 0;

  for (++block_it; block_it != blocks_.end(); ++block_it) {
    if (block_it->num_samples() > count) {
      break;
    }
    num_samples_dropped += block_it->num_samples();
    count -= block_it->num_samples();
  }

  blocks_.erase(blocks_.begin(), block_it);  // Drop whole blocks.
  if (!blocks_.empty()) {
    first_block_offset_ = count;  // Drop part of the next block.
    num_samples_dropped += count;
  }

  num_samples_ -= num_samples_dropped;
  return num_samples_dropped;
}

absl::Status TimeSeriesFramerCalculator::Process(CalculatorContext* cc) {
  if (initial_input_timestamp_ == Timestamp::Unstarted()) {
    initial_input_timestamp_ = cc->InputTimestamp();
    current_timestamp_ = initial_input_timestamp_;
  }

  // Add input data to the internal buffer.
  sample_buffer_.Push(cc->Inputs().Index(0).Get<Matrix>(),
                      cc->InputTimestamp());

  // Construct and emit framed output packets.
  while (sample_buffer_.num_samples() >=
         frame_duration_samples_ + samples_still_to_drop_) {
    sample_buffer_.DropSamples(samples_still_to_drop_);
    Matrix output_frame = sample_buffer_.CopySamples(frame_duration_samples_,
                                                     &current_timestamp_);
    const int frame_step_samples = next_frame_step_samples();
    samples_still_to_drop_ = frame_step_samples;

    if (use_window_) {
      // Apply the window to each row of output_frame.
      output_frame.array().rowwise() *= window_.array();
    }

    cc->Outputs().Index(0).AddPacket(MakePacket<Matrix>(std::move(output_frame))
                                         .At(CurrentOutputTimestamp()));
    ++cumulative_output_frames_;
    cumulative_completed_samples_ += frame_step_samples;
  }
  if (!use_local_timestamp_) {
    // In non-local timestamp mode the timestamp of the next packet will be
    // equal to CumulativeOutputTimestamp(). Inform the framework about this
    // fact to enable packet queueing optimizations.
    cc->Outputs().Index(0).SetNextTimestampBound(CumulativeOutputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status TimeSeriesFramerCalculator::Close(CalculatorContext* cc) {
  sample_buffer_.DropSamples(samples_still_to_drop_);

  if (sample_buffer_.num_samples() > 0 && pad_final_packet_) {
    Matrix output_frame = sample_buffer_.CopySamples(frame_duration_samples_,
                                                     &current_timestamp_);
    cc->Outputs().Index(0).AddPacket(MakePacket<Matrix>(std::move(output_frame))
                                         .At(CurrentOutputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status TimeSeriesFramerCalculator::Open(CalculatorContext* cc) {
  TimeSeriesFramerCalculatorOptions framer_options =
      cc->Options<TimeSeriesFramerCalculatorOptions>();

  RET_CHECK_GT(framer_options.frame_duration_seconds(), 0.0)
      << "Invalid or missing frame_duration_seconds. "
      << "framer_duration_seconds: \n"
      << framer_options.frame_duration_seconds();
  RET_CHECK_LT(framer_options.frame_overlap_seconds(),
               framer_options.frame_duration_seconds())
      << "Invalid frame_overlap_seconds. framer_overlap_seconds: \n"
      << framer_options.frame_overlap_seconds();

  TimeSeriesHeader input_header;
  MP_RETURN_IF_ERROR(time_series_util::FillTimeSeriesHeaderIfValid(
      cc->Inputs().Index(0).Header(), &input_header));

  sample_rate_ = input_header.sample_rate();
  sample_buffer_.Init(sample_rate_, input_header.num_channels());
  frame_duration_samples_ = time_series_util::SecondsToSamples(
      framer_options.frame_duration_seconds(), sample_rate_);
  RET_CHECK_GT(frame_duration_samples_, 0)
      << "Frame duration of " << framer_options.frame_duration_seconds()
      << "s too small to cover a single sample at " << sample_rate_ << " Hz ";
  if (framer_options.emulate_fractional_frame_overlap()) {
    // Frame step may be fractional.
    average_frame_step_samples_ = (framer_options.frame_duration_seconds() -
                                   framer_options.frame_overlap_seconds()) *
                                  sample_rate_;
  } else {
    // Frame step is an integer (stored in a double).
    average_frame_step_samples_ =
        frame_duration_samples_ -
        time_series_util::SecondsToSamples(
            framer_options.frame_overlap_seconds(), sample_rate_);
  }
  RET_CHECK_GE(average_frame_step_samples_, 1)
      << "Frame step too small to cover a single sample at " << sample_rate_
      << " Hz.";
  pad_final_packet_ = framer_options.pad_final_packet();

  auto output_header = new TimeSeriesHeader(input_header);
  output_header->set_num_samples(frame_duration_samples_);
  if (round(average_frame_step_samples_) == average_frame_step_samples_) {
    // Only set output packet rate if it is fixed.
    output_header->set_packet_rate(sample_rate_ / average_frame_step_samples_);
  }
  cc->Outputs().Index(0).SetHeader(Adopt(output_header));
  cumulative_completed_samples_ = 0;
  cumulative_output_frames_ = 0;
  samples_still_to_drop_ = 0;
  initial_input_timestamp_ = Timestamp::Unstarted();
  current_timestamp_ = Timestamp::Unstarted();

  std::vector<double> window_vector;
  use_window_ = false;
  switch (framer_options.window_function()) {
    case TimeSeriesFramerCalculatorOptions::HAMMING:
      audio_dsp::HammingWindow().GetPeriodicSamples(frame_duration_samples_,
                                                    &window_vector);
      use_window_ = true;
      break;
    case TimeSeriesFramerCalculatorOptions::HANN:
      audio_dsp::HannWindow().GetPeriodicSamples(frame_duration_samples_,
                                                 &window_vector);
      use_window_ = true;
      break;
    case TimeSeriesFramerCalculatorOptions::NONE:
      break;
  }

  if (use_window_) {
    window_ = Eigen::Map<Eigen::RowVectorXd>(window_vector.data(),
                                             frame_duration_samples_)
                  .cast<float>();
  }
  use_local_timestamp_ = framer_options.use_local_timestamp();

  return absl::OkStatus();
}

}  // namespace mediapipe
