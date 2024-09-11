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
// Defines SpectrogramCalculator.
#include <math.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "audio/dsp/spectrogram/spectrogram.h"
#include "audio/dsp/window_functions.h"
#include "mediapipe/calculators/audio/spectrogram_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

namespace {
constexpr char kFrameDurationTag[] = "FRAME_DURATION";
constexpr char kFrameOverlapTag[] = "FRAME_OVERLAP";
}  // namespace
// MediaPipe Calculator for computing the "spectrogram" (short-time Fourier
// transform squared-magnitude, by default) of a multichannel input
// time series, including optionally overlapping frames.  Options are
// specified in SpectrogramCalculatorOptions proto  (where names are chosen
// to mirror TimeSeriesFramerCalculator):
//
// Result is a MatrixData record (for single channel input and when the
// allow_multichannel_input flag is false), or a vector of MatrixData records,
// one for each channel (when the allow_multichannel_input flag is set). Each
// waveform frame is converted to frequency by a fast Fourier transform whose
// size, n_fft, is the smallest power of two large enough to enclose the frame
// length of round(frame_duration_seconds * sample_rate).The rows of each
// spectrogram matrix(result) correspond to the n_fft/2+1 unique complex values,
// or squared/linear/dB magnitudes, depending on the output_type option. Each
// input packet will result in zero or one output packets, each containing one
// Matrix for each channel of the input, where each Matrix has one or more
// columns of spectral values, one for each complete frame of input samples. If
// the input packet contains too few samples to trigger a new output frame, no
// output packet is generated (since zero-length packets are not legal since
// they would result in timestamps that were equal, not strictly increasing).
//
// Output packet Timestamps are set to the beginning of each frame. This is to
// allow calculators downstream from SpectrogramCalculator to have aligned
// Timestamps regardless of a packet's signal length.
//
// Both frame_duration_seconds and frame_overlap_seconds will be
// rounded to the nearest integer number of samples. Consequently, all output
// frames will be based on the same number of input samples, and each
// analysis frame will advance from its predecessor by the same time step.
class SpectrogramCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Matrix>(
        // Input stream with TimeSeriesHeader.
    );

    if (cc->InputSidePackets().HasTag(kFrameDurationTag)) {
      cc->InputSidePackets()
          .Tag(kFrameDurationTag)
          .Set<double>(
              // Optional side packet for frame_duration_seconds if provided.
          );
    }

    if (cc->InputSidePackets().HasTag(kFrameOverlapTag)) {
      cc->InputSidePackets()
          .Tag(kFrameOverlapTag)
          .Set<double>(
              // Optional side packet for frame_overlap_seconds if provided.
          );
    }

    SpectrogramCalculatorOptions spectrogram_options =
        cc->Options<SpectrogramCalculatorOptions>();
    if (!spectrogram_options.allow_multichannel_input()) {
      if (spectrogram_options.output_type() ==
          SpectrogramCalculatorOptions::COMPLEX) {
        cc->Outputs().Index(0).Set<Eigen::MatrixXcf>(
            // Complex spectrogram frames with TimeSeriesHeader.
        );
      } else {
        cc->Outputs().Index(0).Set<Matrix>(
            // Spectrogram frames with TimeSeriesHeader.
        );
      }
    } else {
      if (spectrogram_options.output_type() ==
          SpectrogramCalculatorOptions::COMPLEX) {
        cc->Outputs().Index(0).Set<std::vector<Eigen::MatrixXcf>>(
            // Complex spectrogram frames with MultiStreamTimeSeriesHeader.
        );
      } else {
        cc->Outputs().Index(0).Set<std::vector<Matrix>>(
            // Spectrogram frames with MultiStreamTimeSeriesHeader.
        );
      }
    }
    return absl::OkStatus();
  }

  // Returns FAIL if the input stream header is invalid.
  absl::Status Open(CalculatorContext* cc) override;

  // Outputs at most one packet consisting of a single Matrix with one or
  // more columns containing the spectral values from as many input frames
  // as are completed by the input samples.  Always returns OK.
  absl::Status Process(CalculatorContext* cc) override;

  // Performs zero-padding and processing of any remaining samples
  // if pad_final_packet is set.
  // Returns OK.
  absl::Status Close(CalculatorContext* cc) override;

 private:
  Timestamp CurrentOutputTimestamp(CalculatorContext* cc) {
    if (use_local_timestamp_) {
      const Timestamp now = cc->InputTimestamp();
      if (now == Timestamp::Done()) {
        // During Close the timestamp is not available, send an estimate.
        return last_local_output_timestamp_ +
               round(last_completed_frames_ * frame_step_samples() *
                     Timestamp::kTimestampUnitsPerSecond / input_sample_rate_);
      }
      last_local_output_timestamp_ = now;
      return now;
    }
    return CumulativeOutputTimestamp();
  }

  Timestamp CumulativeOutputTimestamp() {
    // Cumulative output timestamp is the *center* of the next frame to be
    // emitted, hence delayed by half a window duration compared to relevant
    // input timestamp.
    return initial_input_timestamp_ +
           round(cumulative_completed_frames_ * frame_step_samples() *
                 Timestamp::kTimestampUnitsPerSecond / input_sample_rate_);
  }

  int frame_step_samples() const {
    return frame_duration_samples_ - frame_overlap_samples_;
  }

  // Take the next set of input samples, already translated into a
  // vector<float> and pass them to the spectrogram object.
  // Convert the output of the spectrogram object into a Matrix (or an
  // Eigen::MatrixXcf if complex-valued output is requested) and pass to
  // MediaPipe output.
  absl::Status ProcessVector(const Matrix& input_stream, CalculatorContext* cc);

  // Templated function to process either real- or complex-output spectrogram.
  template <class OutputMatrixType>
  absl::Status ProcessVectorToOutput(
      const Matrix& input_stream,
      const OutputMatrixType postprocess_output_fn(const OutputMatrixType&),
      CalculatorContext* cc);

  // Use the MediaPipe timestamp instead of the estimated one. Useful when the
  // data is intermittent.
  bool use_local_timestamp_;
  Timestamp last_local_output_timestamp_;

  double input_sample_rate_;
  bool pad_final_packet_;
  int frame_duration_samples_;
  int frame_overlap_samples_;
  // How many samples we've been passed, used for checking input time stamps.
  int64_t cumulative_input_samples_;
  // How many frames we've emitted, used for calculating output time stamps.
  int64_t cumulative_completed_frames_;
  // How many frames were emitted last, used for estimating the timestamp on
  // Close when use_local_timestamp_ is true;
  int64_t last_completed_frames_;
  Timestamp initial_input_timestamp_;
  int num_input_channels_;
  // How many frequency bins we emit (=N_FFT/2 + 1).
  int num_output_channels_;
  // Which output type?
  int output_type_;
  // Output type: mono or multichannel.
  bool allow_multichannel_input_;
  // Vector of Spectrogram objects, one for each channel.
  std::vector<std::unique_ptr<audio_dsp::Spectrogram>> spectrogram_generators_;
  // Whether to reset the Spectrogram sample buffer on every call to Process.
  bool reset_sample_buffer_;
  // Fixed scale factor applied to input values.
  float input_scale_;
  // Fixed scale factor applied to output values (regardless of type).
  double output_scale_;

  static const float kLnSquaredMagnitudeToDb;
};
REGISTER_CALCULATOR(SpectrogramCalculator);

// DECIBELS = 20*log10(LINEAR_MAGNITUDE) = 10*Log10(SQUARED_MAGNITUDE)
// =10/ln(10)*ln(SQUARED_MAGNITUDE).
// Factor to convert ln(SQUARED_MAGNITUDE) to deciBels = 10.0/ln(10.0).
const float SpectrogramCalculator::kLnSquaredMagnitudeToDb = 4.342944819032518;

namespace {
std::unique_ptr<audio_dsp::WindowFunction> MakeWindowFun(
    const SpectrogramCalculatorOptions::WindowType window_type) {
  switch (window_type) {
    // The cosine window and square root of Hann are equivalent.
    case SpectrogramCalculatorOptions::COSINE:
    case SpectrogramCalculatorOptions::SQRT_HANN:
      return std::make_unique<audio_dsp::CosineWindow>();
    case SpectrogramCalculatorOptions::HANN:
      return std::make_unique<audio_dsp::HannWindow>();
    case SpectrogramCalculatorOptions::HAMMING:
      return std::make_unique<audio_dsp::HammingWindow>();
  }
  return nullptr;
}
}  // namespace

absl::Status SpectrogramCalculator::Open(CalculatorContext* cc) {
  SpectrogramCalculatorOptions spectrogram_options =
      cc->Options<SpectrogramCalculatorOptions>();
  // Provide frame_duration_seconds and frame_overlap_seconds either from static
  // options, or dynamically from a side packet, the side packet one will
  // override the options one if provided.

  double frame_duration_seconds = 0;
  double frame_overlap_seconds = 0;
  if (cc->InputSidePackets().HasTag(kFrameDurationTag)) {
    frame_duration_seconds =
        cc->InputSidePackets().Tag(kFrameDurationTag).Get<double>();
  } else {
    frame_duration_seconds = spectrogram_options.frame_duration_seconds();
  }

  if (cc->InputSidePackets().HasTag(kFrameOverlapTag)) {
    frame_overlap_seconds =
        cc->InputSidePackets().Tag(kFrameOverlapTag).Get<double>();
  } else {
    frame_overlap_seconds = spectrogram_options.frame_overlap_seconds();
  }

  use_local_timestamp_ = spectrogram_options.use_local_timestamp();

  if (frame_duration_seconds <= 0.0) {
    // TODO: return an error.
  }
  if (frame_overlap_seconds >= frame_duration_seconds) {
    // TODO: return an error.
  }
  if (frame_overlap_seconds < 0.0) {
    // TODO: return an error.
  }

  TimeSeriesHeader input_header;
  MP_RETURN_IF_ERROR(time_series_util::FillTimeSeriesHeaderIfValid(
      cc->Inputs().Index(0).Header(), &input_header));

  input_sample_rate_ = input_header.sample_rate();
  num_input_channels_ = input_header.num_channels();

  if (!spectrogram_options.allow_multichannel_input() &&
      num_input_channels_ != 1) {
    // TODO: return an error.
  }

  frame_duration_samples_ = round(frame_duration_seconds * input_sample_rate_);
  frame_overlap_samples_ = round(frame_overlap_seconds * input_sample_rate_);

  pad_final_packet_ = spectrogram_options.pad_final_packet();
  output_type_ = spectrogram_options.output_type();
  allow_multichannel_input_ = spectrogram_options.allow_multichannel_input();

  input_scale_ = spectrogram_options.input_scale();
  output_scale_ = spectrogram_options.output_scale();

  auto window_fun = MakeWindowFun(spectrogram_options.window_type());
  if (window_fun == nullptr) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrCat("Invalid window type ",
                                     spectrogram_options.window_type()));
  }
  std::vector<double> window;

  window_fun->GetPeriodicSamples(frame_duration_samples_, &window);

  // Propagate settings down to the actual Spectrogram object.
  std::optional<int> fft_size;
  if (spectrogram_options.fft_size() > 0) {
    fft_size = spectrogram_options.fft_size();
  }

  spectrogram_generators_.clear();
  for (int i = 0; i < num_input_channels_; i++) {
    spectrogram_generators_.push_back(
        std::make_unique<audio_dsp::Spectrogram>());
    spectrogram_generators_[i]->Initialize(window, frame_step_samples(),
                                           fft_size);
  }

  switch (spectrogram_options.sample_buffer_mode()) {
    case SpectrogramCalculatorOptions::NONE:
      reset_sample_buffer_ = false;
      break;
    case SpectrogramCalculatorOptions::RESET:
      reset_sample_buffer_ = true;
      break;
    default:
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          "Unrecognized spectrogram sample buffer mode.");
  }

  num_output_channels_ =
      spectrogram_generators_[0]->output_frequency_channels();
  std::unique_ptr<TimeSeriesHeader> output_header(
      new TimeSeriesHeader(input_header));
  // Store the actual sample rate of the input audio in the TimeSeriesHeader
  // so that subsequent calculators can figure out the frequency scale of
  // our output.
  output_header->set_audio_sample_rate(input_sample_rate_);
  // Setup rest of output header.
  output_header->set_num_channels(num_output_channels_);
  output_header->set_sample_rate(input_sample_rate_ / frame_step_samples());
  // Although we usually generate one output packet for each input
  // packet, this might not be true for input packets whose size is smaller
  // than the analysis window length.  So we clear output_header.packet_rate
  // because we can't guarantee a constant packet rate.  Similarly, the number
  // of output frames per packet depends on the input packet, so we also clear
  // output_header.num_samples.
  output_header->clear_packet_rate();
  output_header->clear_num_samples();
  if (!spectrogram_options.allow_multichannel_input()) {
    cc->Outputs().Index(0).SetHeader(Adopt(output_header.release()));
  } else {
    std::unique_ptr<MultiStreamTimeSeriesHeader> multichannel_output_header(
        new MultiStreamTimeSeriesHeader());
    *multichannel_output_header->mutable_time_series_header() = *output_header;
    multichannel_output_header->set_num_streams(num_input_channels_);
    cc->Outputs().Index(0).SetHeader(
        Adopt(multichannel_output_header.release()));
  }
  cumulative_completed_frames_ = 0;
  last_completed_frames_ = 0;
  initial_input_timestamp_ = Timestamp::Unstarted();
  if (use_local_timestamp_) {
    // Inform the framework that the calculator will output packets at the same
    // timestamps as input packets to enable packet queueing optimizations. The
    // final packet (emitted from Close()) does not follow this rule but it's
    // sufficient that its timestamp is strictly greater than the timestamp of
    // the previous packet.
    cc->SetOffset(0);
  }
  return absl::OkStatus();
}

absl::Status SpectrogramCalculator::Process(CalculatorContext* cc) {
  if (initial_input_timestamp_ == Timestamp::Unstarted()) {
    initial_input_timestamp_ = cc->InputTimestamp();
  }

  const Matrix& input_stream = cc->Inputs().Index(0).Get<Matrix>();
  if (input_stream.rows() != num_input_channels_) {
    // TODO: return an error.
  }

  cumulative_input_samples_ += input_stream.cols();

  return ProcessVector(input_stream, cc);
}

template <class OutputMatrixType>
absl::Status SpectrogramCalculator::ProcessVectorToOutput(
    const Matrix& input_stream,
    const OutputMatrixType postprocess_output_fn(const OutputMatrixType&),
    CalculatorContext* cc) {
  std::unique_ptr<std::vector<OutputMatrixType>> spectrogram_matrices(
      new std::vector<OutputMatrixType>());
  std::vector<std::vector<typename OutputMatrixType::Scalar>> output_vectors;

  // Compute a spectrogram for each channel.
  int num_output_time_frames;
  for (int channel = 0; channel < input_stream.rows(); ++channel) {
    output_vectors.clear();

    // Copy one row (channel) of the input matrix into the std::vector.
    std::vector<float> input_vector(input_stream.cols());
    Eigen::Map<Matrix>(&input_vector[0], 1, input_vector.size()) =
        input_stream.row(channel) * input_scale_;

    if (reset_sample_buffer_) {
      spectrogram_generators_[channel]->ResetSampleBuffer();
    }
    if (!spectrogram_generators_[channel]->ComputeSpectrogram(
            input_vector, &output_vectors)) {
      return absl::Status(absl::StatusCode::kInternal,
                          "Spectrogram returned failure");
    }
    if (channel == 0) {
      // Record the number of time frames we expect from each channel.
      num_output_time_frames = output_vectors.size();
    } else {
      RET_CHECK_EQ(output_vectors.size(), num_output_time_frames)
          << "Inconsistent spectrogram time frames for channel " << channel;
    }
    // Skip remaining processing if there are too few input samples to trigger
    // any output frames.
    if (!output_vectors.empty()) {
      // Translate the returned values into a matrix of output frames.
      OutputMatrixType output_frames(num_output_channels_,
                                     output_vectors.size());
      for (int frame = 0; frame < output_vectors.size(); ++frame) {
        Eigen::Map<const OutputMatrixType> frame_map(
            &output_vectors[frame][0], output_vectors[frame].size(), 1);
        // The underlying dsp object returns squared magnitudes; here
        // we optionally translate to linear magnitude or dB.
        output_frames.col(frame) =
            output_scale_ * postprocess_output_fn(frame_map);
      }
      spectrogram_matrices->push_back(output_frames);
    }
  }
  // If the input is very short, there may not be enough accumulated,
  // unprocessed samples to cause any new frames to be generated by
  // the spectrogram object.  If so, we don't want to emit
  // a packet at all.
  if (!spectrogram_matrices->empty()) {
    RET_CHECK_EQ(spectrogram_matrices->size(), input_stream.rows())
        << "Inconsistent number of spectrogram channels.";
    if (allow_multichannel_input_) {
      cc->Outputs().Index(0).Add(spectrogram_matrices.release(),
                                 CurrentOutputTimestamp(cc));
    } else {
      cc->Outputs().Index(0).Add(
          new OutputMatrixType(spectrogram_matrices->at(0)),
          CurrentOutputTimestamp(cc));
    }
    cumulative_completed_frames_ += output_vectors.size();
    last_completed_frames_ = output_vectors.size();
    if (!use_local_timestamp_) {
      // In non-local timestamp mode the timestamp of the next packet will be
      // equal to CumulativeOutputTimestamp(). Inform the framework about this
      // fact to enable packet queueing optimizations.
      cc->Outputs().Index(0).SetNextTimestampBound(CumulativeOutputTimestamp());
    }
  }
  return absl::OkStatus();
}

absl::Status SpectrogramCalculator::ProcessVector(const Matrix& input_stream,
                                                  CalculatorContext* cc) {
  switch (output_type_) {
    // These blocks deliberately ignore clang-format to preserve the
    // "silhouette" of the different cases.
    // clang-format off
    case SpectrogramCalculatorOptions::COMPLEX: {
      return ProcessVectorToOutput(
          input_stream,
          +[](const Eigen::MatrixXcf& col) -> const Eigen::MatrixXcf {
            return col;
          }, cc);
    }
    case SpectrogramCalculatorOptions::SQUARED_MAGNITUDE: {
      return ProcessVectorToOutput(
          input_stream,
          +[](const Matrix& col) -> const Matrix {
            return col;
          }, cc);
    }
    case SpectrogramCalculatorOptions::LINEAR_MAGNITUDE: {
      return ProcessVectorToOutput(
          input_stream,
          +[](const Matrix& col) -> const Matrix {
            return col.array().sqrt().matrix();
          }, cc);
    }
    case SpectrogramCalculatorOptions::DECIBELS: {
      return ProcessVectorToOutput(
          input_stream,
          +[](const Matrix& col) -> const Matrix {
            return kLnSquaredMagnitudeToDb * col.array().log().matrix();
          }, cc);
    }
    // clang-format on
    default: {
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          "Unrecognized spectrogram output type.");
    }
  }
}

absl::Status SpectrogramCalculator::Close(CalculatorContext* cc) {
  if (cumulative_input_samples_ > 0 && pad_final_packet_) {
    // We can flush any remaining samples by sending frame_step_samples - 1
    // zeros to the Process method, and letting it do its thing,
    // UNLESS we have fewer than one window's worth of samples, in which case
    // we pad to exactly one frame_duration_samples.
    // Release the memory for the Spectrogram objects.
    int required_padding_samples = frame_step_samples() - 1;
    if (cumulative_input_samples_ < frame_duration_samples_) {
      required_padding_samples =
          frame_duration_samples_ - cumulative_input_samples_;
    }
    return ProcessVector(
        Matrix::Zero(num_input_channels_, required_padding_samples), cc);
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
