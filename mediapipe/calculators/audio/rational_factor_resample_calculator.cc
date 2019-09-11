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
// Defines RationalFactorResampleCalculator.

#include "mediapipe/calculators/audio/rational_factor_resample_calculator.h"

#include "audio/dsp/resampler_rational_factor.h"

using audio_dsp::DefaultResamplingKernel;
using audio_dsp::RationalFactorResampler;
using audio_dsp::Resampler;

namespace mediapipe {
::mediapipe::Status RationalFactorResampleCalculator::Process(
    CalculatorContext* cc) {
  return ProcessInternal(cc->Inputs().Index(0).Get<Matrix>(), false, cc);
}

::mediapipe::Status RationalFactorResampleCalculator::Close(
    CalculatorContext* cc) {
  if (initial_timestamp_ == Timestamp::Unstarted()) {
    return ::mediapipe::OkStatus();
  }
  Matrix empty_input_frame(num_channels_, 0);
  return ProcessInternal(empty_input_frame, true, cc);
}

namespace {
void CopyChannelToVector(const Matrix& matrix, int channel,
                         std::vector<float>* vec) {
  vec->clear();
  vec->reserve(matrix.cols());
  for (int sample = 0; sample < matrix.cols(); ++sample) {
    vec->push_back(matrix(channel, sample));
  }
}

void CopyVectorToChannel(const std::vector<float>& vec, Matrix* matrix,
                         int channel) {
  if (matrix->cols() == 0) {
    matrix->resize(matrix->rows(), vec.size());
  } else {
    CHECK_EQ(vec.size(), matrix->cols());
    CHECK_LT(channel, matrix->rows());
  }
  for (int sample = 0; sample < matrix->cols(); ++sample) {
    (*matrix)(channel, sample) = vec[sample];
  }
}

}  // namespace

::mediapipe::Status RationalFactorResampleCalculator::Open(
    CalculatorContext* cc) {
  RationalFactorResampleCalculatorOptions resample_options =
      cc->Options<RationalFactorResampleCalculatorOptions>();

  if (!resample_options.has_target_sample_rate()) {
    return tool::StatusInvalid(
        "resample_options doesn't have target_sample_rate.");
  }
  target_sample_rate_ = resample_options.target_sample_rate();

  TimeSeriesHeader input_header;
  MP_RETURN_IF_ERROR(time_series_util::FillTimeSeriesHeaderIfValid(
      cc->Inputs().Index(0).Header(), &input_header));

  source_sample_rate_ = input_header.sample_rate();
  num_channels_ = input_header.num_channels();

  // Don't create resamplers for pass-thru (sample rates are equal).
  if (source_sample_rate_ != target_sample_rate_) {
    resampler_.resize(num_channels_);
    for (auto& r : resampler_) {
      r = ResamplerFromOptions(source_sample_rate_, target_sample_rate_,
                               resample_options);
      if (!r) {
        LOG(ERROR) << "Failed to initialize resampler.";
        return ::mediapipe::UnknownError("Failed to initialize resampler.");
      }
    }
  }

  TimeSeriesHeader* output_header = new TimeSeriesHeader(input_header);
  output_header->set_sample_rate(target_sample_rate_);
  // The resampler doesn't make guarantees about how many samples will
  // be in each packet.
  output_header->clear_packet_rate();
  output_header->clear_num_samples();

  cc->Outputs().Index(0).SetHeader(Adopt(output_header));
  cumulative_output_samples_ = 0;
  cumulative_input_samples_ = 0;
  initial_timestamp_ = Timestamp::Unstarted();
  check_inconsistent_timestamps_ =
      resample_options.check_inconsistent_timestamps();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status RationalFactorResampleCalculator::ProcessInternal(
    const Matrix& input_frame, bool should_flush, CalculatorContext* cc) {
  if (initial_timestamp_ == Timestamp::Unstarted()) {
    initial_timestamp_ = cc->InputTimestamp();
  }

  if (check_inconsistent_timestamps_) {
    time_series_util::LogWarningIfTimestampIsInconsistent(
        cc->InputTimestamp(), initial_timestamp_, cumulative_input_samples_,
        source_sample_rate_);
  }
  Timestamp output_timestamp =
      initial_timestamp_ + ((cumulative_output_samples_ / target_sample_rate_) *
                            Timestamp::kTimestampUnitsPerSecond);

  cumulative_input_samples_ += input_frame.cols();
  std::unique_ptr<Matrix> output_frame(new Matrix(num_channels_, 0));
  if (resampler_.empty()) {
    // Sample rates were same for input and output; pass-thru.
    *output_frame = input_frame;
  } else {
    if (!Resample(input_frame, output_frame.get(), should_flush)) {
      return ::mediapipe::UnknownError("Resample() failed.");
    }
  }
  cumulative_output_samples_ += output_frame->cols();

  if (output_frame->cols() > 0) {
    cc->Outputs().Index(0).Add(output_frame.release(), output_timestamp);
  }
  return ::mediapipe::OkStatus();
}

bool RationalFactorResampleCalculator::Resample(const Matrix& input_frame,
                                                Matrix* output_frame,
                                                bool should_flush) {
  std::vector<float> input_vector;
  std::vector<float> output_vector;
  for (int i = 0; i < input_frame.rows(); ++i) {
    CopyChannelToVector(input_frame, i, &input_vector);
    if (should_flush) {
      resampler_[i]->Flush(&output_vector);
    } else {
      resampler_[i]->ProcessSamples(input_vector, &output_vector);
    }
    CopyVectorToChannel(output_vector, output_frame, i);
  }
  return true;
}

// static
std::unique_ptr<Resampler<float>>
RationalFactorResampleCalculator::ResamplerFromOptions(
    const double source_sample_rate, const double target_sample_rate,
    const RationalFactorResampleCalculatorOptions& options) {
  std::unique_ptr<Resampler<float>> resampler;
  const auto& rational_factor_options =
      options.resampler_rational_factor_options();
  std::unique_ptr<DefaultResamplingKernel> kernel;
  if (rational_factor_options.has_radius() &&
      rational_factor_options.has_cutoff() &&
      rational_factor_options.has_kaiser_beta()) {
    kernel = absl::make_unique<DefaultResamplingKernel>(
        source_sample_rate, target_sample_rate,
        rational_factor_options.radius(), rational_factor_options.cutoff(),
        rational_factor_options.kaiser_beta());
  } else {
    kernel = absl::make_unique<DefaultResamplingKernel>(source_sample_rate,
                                                        target_sample_rate);
  }

  // Set large enough so that the resampling factor between common sample
  // rates (e.g. 8kHz, 16kHz, 22.05kHz, 32kHz, 44.1kHz, 48kHz) is exact, and
  // that any factor is represented with error less than 0.025%.
  const int kMaxDenominator = 2000;
  resampler = absl::make_unique<RationalFactorResampler<float>>(
      *kernel, kMaxDenominator);
  if (resampler != nullptr && !resampler->Valid()) {
    resampler = std::unique_ptr<Resampler<float>>();
  }
  return resampler;
}

REGISTER_CALCULATOR(RationalFactorResampleCalculator);

}  // namespace mediapipe
