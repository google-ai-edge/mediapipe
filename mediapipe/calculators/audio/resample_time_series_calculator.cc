// Copyright 2025 The MediaPipe Authors.
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

#include "mediapipe/calculators/audio/resample_time_series_calculator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "audio/dsp/resampler_q.h"
#include "mediapipe/calculators/audio/resample_time_series_calculator.pb.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

namespace {

void CopyChannelToVector(const mediapipe::Matrix& matrix, int channel,
                         std::vector<float>* vec) {
  vec->resize(matrix.cols());
  Eigen::Map<Eigen::ArrayXf>(vec->data(), vec->size()) = matrix.row(channel);
}

void CopyVectorToChannel(const std::vector<float>& vec,
                         mediapipe::Matrix* matrix, int channel) {
  if (matrix->cols() == 0) {
    matrix->resize(matrix->rows(), vec.size());
  } else {
    ABSL_CHECK_EQ(vec.size(), matrix->cols());
  }
  ABSL_CHECK_LT(channel, matrix->rows());
  matrix->row(channel) =
      Eigen::Map<const Eigen::ArrayXf>(vec.data(), vec.size());
}

Timestamp CalculateOutputTimestamp(Timestamp initial_timestamp,
                                   int64_t cumulative_output_samples,
                                   double target_sample_rate) {
  ABSL_DCHECK(initial_timestamp != Timestamp::Unstarted());
  return initial_timestamp + ((cumulative_output_samples / target_sample_rate) *
                              Timestamp::kTimestampUnitsPerSecond);
}

}  // namespace

// Defines ResampleTimeSeriesCalculator.
absl::Status ResampleTimeSeriesCalculatorImpl::Process(CalculatorContext* cc) {
  return ProcessInternal(cc, kInput(cc).Get(), false);
}

absl::Status ResampleTimeSeriesCalculatorImpl::Close(CalculatorContext* cc) {
  if (initial_timestamp_ == Timestamp::Unstarted()) {
    return absl::OkStatus();
  }
  Matrix empty_input_frame(num_channels_, 0);
  return ProcessInternal(cc, empty_input_frame, true);
}

absl::Status ResampleTimeSeriesCalculatorImpl::Open(CalculatorContext* cc) {
  ResampleTimeSeriesCalculatorOptions resample_options;
  time_series_util::FillOptionsExtensionOrDie(cc->Options(), &resample_options);

  // Provide target_sample_rate either from static options, or dynamically from
  // a side packet, the side packet one will override the options one if
  // provided.
  if (resample_options.has_target_sample_rate()) {
    target_sample_rate_ = resample_options.target_sample_rate();
  } else if (!kSideInputTargetSampleRate(cc).IsEmpty()) {
    target_sample_rate_ = kSideInputTargetSampleRate(cc).Get();
  } else {
    return tool::StatusInvalid(
        "target_sample_rate is not provided in resample_options, nor from a "
        "side packet.");
  }

  double min_source_sample_rate = target_sample_rate_;
  if (resample_options.allow_upsampling()) {
    min_source_sample_rate = resample_options.min_source_sample_rate();
  }

  TimeSeriesHeader input_header;
  MP_RETURN_IF_ERROR(time_series_util::FillTimeSeriesHeaderIfValid(
      kInput(cc).Header(), &input_header));

  source_sample_rate_ = input_header.sample_rate();
  num_channels_ = input_header.num_channels();

  if (source_sample_rate_ < min_source_sample_rate) {
    return ::absl::FailedPreconditionError(
        "Resample() failed because upsampling is disabled or source sample "
        "rate is lower than min_source_sample_rate.");
  }

  // Don't create resamplers for pass-thru (sample rates are equal).
  if (source_sample_rate_ != target_sample_rate_) {
    resampler_ = ResamplerFromOptions(source_sample_rate_, target_sample_rate_,
                                      num_channels_, resample_options);
    RET_CHECK(resampler_) << "Failed to initialize resampler.";
  }

  TimeSeriesHeader* output_header = new TimeSeriesHeader(input_header);
  output_header->set_sample_rate(target_sample_rate_);
  // The resampler doesn't make guarantees about how many samples will
  // be in each packet.
  output_header->clear_packet_rate();
  output_header->clear_num_samples();

  kOutput(cc).SetHeader(mediapipe::api2::FromOldPacket(Adopt(output_header)));
  cumulative_output_samples_ = 0;
  cumulative_input_samples_ = 0;
  initial_timestamp_ = Timestamp::Unstarted();
  check_inconsistent_timestamps_ =
      resample_options.check_inconsistent_timestamps();
  return absl::OkStatus();
}

absl::Status ResampleTimeSeriesCalculatorImpl::ProcessInternal(
    CalculatorContext* cc, const Matrix& input_frame, bool should_flush) {
  if (initial_timestamp_ == Timestamp::Unstarted()) {
    initial_timestamp_ = kInput(cc).timestamp();
  }

  if (check_inconsistent_timestamps_) {
    time_series_util::LogWarningIfTimestampIsInconsistent(
        kInput(cc).timestamp(), initial_timestamp_, cumulative_input_samples_,
        source_sample_rate_);
  }
  const Timestamp output_timestamp = CalculateOutputTimestamp(
      initial_timestamp_, cumulative_output_samples_, target_sample_rate_);

  cumulative_input_samples_ += input_frame.cols();
  std::unique_ptr<Matrix> output_frame(new Matrix(num_channels_, 0));
  if (resampler_ == nullptr) {
    // Sample rates were same for input and output; pass-thru.
    *output_frame = input_frame;
  } else {
    resampler_->Resample(input_frame, output_frame.get(), should_flush);
  }
  cumulative_output_samples_ += output_frame->cols();

  if (output_frame->cols() > 0) {
    kOutput(cc).Send(*output_frame, output_timestamp);
    output_frame.reset();
  }
  kOutput(cc).SetNextTimestampBound(CalculateOutputTimestamp(
      initial_timestamp_, cumulative_output_samples_, target_sample_rate_));

  return absl::OkStatus();
}

// static
std::unique_ptr<ResampleTimeSeriesCalculatorImpl::ResamplerWrapper>
ResampleTimeSeriesCalculatorImpl::ResamplerFromOptions(
    double source_sample_rate, double target_sample_rate, int num_channels,
    const ResampleTimeSeriesCalculatorOptions& options) {
  std::unique_ptr<ResamplerWrapper> resampler;
  switch (options.resampler_type()) {
    case ResampleTimeSeriesCalculatorOptions::RESAMPLER_RATIONAL_FACTOR: {
      const auto& rational_factor_options =
          options.resampler_rational_factor_options();

      // Read resampler parameters from proto.
      audio_dsp::QResamplerParams params;
      if (rational_factor_options.has_radius_factor()) {
        params.filter_radius_factor = rational_factor_options.radius_factor();
      } else if (rational_factor_options.has_radius()) {
        // Convert RationalFactorResampler radius to QResampler radius_factor.
        params.filter_radius_factor =
            rational_factor_options.radius() *
            std::min(1.0, target_sample_rate / source_sample_rate);
      }
      if (rational_factor_options.has_cutoff_proportion()) {
        params.cutoff_proportion = rational_factor_options.cutoff_proportion();
      } else if (rational_factor_options.has_cutoff()) {
        // Convert RationalFactorResampler cutoff to QResampler
        // cutoff_proportion.
        params.cutoff_proportion =
            2 * rational_factor_options.cutoff() /
            std::min(source_sample_rate, target_sample_rate);
      }
      if (rational_factor_options.has_kaiser_beta()) {
        params.kaiser_beta = rational_factor_options.kaiser_beta();
      }
      // Set large enough so that the resampling factor between common sample
      // rates (e.g. 8kHz, 16kHz, 22.05kHz, 32kHz, 44.1kHz, 48kHz) is exact, and
      // that any factor is represented with error less than 0.025%.
      params.max_denominator = 2000;

      resampler = std::make_unique<QResamplerWrapper>(
          source_sample_rate, target_sample_rate, num_channels, params);
    } break;
    default:
      break;
  }
  if (resampler != nullptr && !resampler->Valid()) {
    resampler.reset();
  }
  return resampler;
}
}  // namespace mediapipe
