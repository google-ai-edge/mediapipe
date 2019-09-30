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

#include <deque>
#include <memory>
#include <string>

#include "Eigen/Core"
#include "audio/dsp/window_functions.h"
#include "mediapipe/calculators/audio/time_series_framer_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
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
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Matrix>(
        // Input stream with TimeSeriesHeader.
    );
    cc->Outputs().Index(0).Set<Matrix>(
        // Fixed length time series Packets with TimeSeriesHeader.
    );
    return ::mediapipe::OkStatus();
  }

  // Returns FAIL if the input stream header is invalid.
  ::mediapipe::Status Open(CalculatorContext* cc) override;

  // Outputs as many framed packets as possible given the accumulated
  // input.  Always returns OK.
  ::mediapipe::Status Process(CalculatorContext* cc) override;

  // Flushes any remaining samples in a zero-padded packet.  Always
  // returns OK.
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  // Adds input data to the internal buffer.
  void EnqueueInput(CalculatorContext* cc);
  // Constructs and emits framed output packets.
  void FrameOutput(CalculatorContext* cc);

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

  // Returns the timestamp of a sample on a base, which is usually the time
  // stamp of a packet.
  Timestamp CurrentSampleTimestamp(const Timestamp& timestamp_base,
                                   int64 number_of_samples) {
    return timestamp_base + round(number_of_samples / sample_rate_ *
                                  Timestamp::kTimestampUnitsPerSecond);
  }

  // The number of input samples to advance after the current output frame is
  // emitted.
  int next_frame_step_samples() const {
    // All numbers are in input samples.
    const int64 current_output_frame_start = static_cast<int64>(
        round(cumulative_output_frames_ * average_frame_step_samples_));
    CHECK_EQ(current_output_frame_start, cumulative_completed_samples_);
    const int64 next_output_frame_start = static_cast<int64>(
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
  int64 cumulative_input_samples_;
  int64 cumulative_output_frames_;
  // "Completed" samples are samples that are no longer needed because
  // the framer has completely stepped past them (taking into account
  // any overlap).
  int64 cumulative_completed_samples_;
  Timestamp initial_input_timestamp_;
  // The current timestamp is updated along with the incoming packets.
  Timestamp current_timestamp_;
  int num_channels_;

  // Each entry in this deque consists of a single sample, i.e. a
  // single column vector, and its timestamp.
  std::deque<std::pair<Matrix, Timestamp>> sample_buffer_;

  bool use_window_;
  Matrix window_;

  bool use_local_timestamp_;
};
REGISTER_CALCULATOR(TimeSeriesFramerCalculator);

void TimeSeriesFramerCalculator::EnqueueInput(CalculatorContext* cc) {
  const Matrix& input_frame = cc->Inputs().Index(0).Get<Matrix>();

  for (int i = 0; i < input_frame.cols(); ++i) {
    sample_buffer_.emplace_back(std::make_pair(
        input_frame.col(i), CurrentSampleTimestamp(cc->InputTimestamp(), i)));
  }

  cumulative_input_samples_ += input_frame.cols();
}

void TimeSeriesFramerCalculator::FrameOutput(CalculatorContext* cc) {
  while (sample_buffer_.size() >=
         frame_duration_samples_ + samples_still_to_drop_) {
    while (samples_still_to_drop_ > 0) {
      sample_buffer_.pop_front();
      --samples_still_to_drop_;
    }
    const int frame_step_samples = next_frame_step_samples();
    std::unique_ptr<Matrix> output_frame(
        new Matrix(num_channels_, frame_duration_samples_));
    for (int i = 0; i < std::min(frame_step_samples, frame_duration_samples_);
         ++i) {
      output_frame->col(i) = sample_buffer_.front().first;
      current_timestamp_ = sample_buffer_.front().second;
      sample_buffer_.pop_front();
    }
    const int frame_overlap_samples =
        frame_duration_samples_ - frame_step_samples;
    if (frame_overlap_samples > 0) {
      for (int i = 0; i < frame_overlap_samples; ++i) {
        output_frame->col(i + frame_step_samples) = sample_buffer_[i].first;
        current_timestamp_ = sample_buffer_[i].second;
      }
    } else {
      samples_still_to_drop_ = -frame_overlap_samples;
    }

    if (use_window_) {
      *output_frame = (output_frame->array() * window_.array()).matrix();
    }

    cc->Outputs().Index(0).Add(output_frame.release(),
                               CurrentOutputTimestamp());
    ++cumulative_output_frames_;
    cumulative_completed_samples_ += frame_step_samples;
  }
}

::mediapipe::Status TimeSeriesFramerCalculator::Process(CalculatorContext* cc) {
  if (initial_input_timestamp_ == Timestamp::Unstarted()) {
    initial_input_timestamp_ = cc->InputTimestamp();
    current_timestamp_ = initial_input_timestamp_;
  }

  EnqueueInput(cc);
  FrameOutput(cc);

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TimeSeriesFramerCalculator::Close(CalculatorContext* cc) {
  while (samples_still_to_drop_ > 0 && !sample_buffer_.empty()) {
    sample_buffer_.pop_front();
    --samples_still_to_drop_;
  }
  if (!sample_buffer_.empty() && pad_final_packet_) {
    std::unique_ptr<Matrix> output_frame(new Matrix);
    output_frame->setZero(num_channels_, frame_duration_samples_);
    for (int i = 0; i < sample_buffer_.size(); ++i) {
      output_frame->col(i) = sample_buffer_[i].first;
      current_timestamp_ = sample_buffer_[i].second;
    }

    cc->Outputs().Index(0).Add(output_frame.release(),
                               CurrentOutputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TimeSeriesFramerCalculator::Open(CalculatorContext* cc) {
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
  num_channels_ = input_header.num_channels();
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
  cumulative_input_samples_ = 0;
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
    window_ = Matrix::Ones(num_channels_, 1) *
              Eigen::Map<Eigen::MatrixXd>(window_vector.data(), 1,
                                          frame_duration_samples_)
                  .cast<float>();
  }
  use_local_timestamp_ = framer_options.use_local_timestamp();

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
