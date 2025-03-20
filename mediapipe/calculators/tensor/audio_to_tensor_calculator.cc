// Copyright 2022 The MediaPipe Authors.
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "audio/dsp/resampler_q.h"
#include "audio/dsp/window_functions.h"
#include "mediapipe/calculators/tensor/audio_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/time_series_util.h"
#include "pffft.h"

namespace mediapipe {
namespace api2 {
namespace {

using Options = ::mediapipe::AudioToTensorCalculatorOptions;
using DftTensorFormat = Options::DftTensorFormat;
using FlushMode = Options::FlushMode;

std::vector<float> HannWindow(int window_size, bool sqrt_hann) {
  std::vector<float> hann_window(window_size);
  audio_dsp::HannWindow().GetPeriodicSamples(window_size, &hann_window);
  if (sqrt_hann) {
    absl::c_transform(hann_window, hann_window.begin(),
                      [](double x) { return std::sqrt(x); });
  }
  return hann_window;
}

// PFFFT only supports transforms for inputs of length N of the form
// N = (2^a)*(3^b)*(5^c) where b >=0 and c >= 0 and a >= 5 for the real FFT.
bool IsValidFftSize(int size) {
  if (size <= 0) {
    return false;
  }
  constexpr int kFactors[] = {2, 3, 5};
  int factorization[] = {0, 0, 0};
  int n = static_cast<int>(size);
  for (int i = 0; i < 3; ++i) {
    while (n % kFactors[i] == 0) {
      n = n / kFactors[i];
      ++factorization[i];
    }
  }
  return factorization[0] >= 5 && n == 1;
}

}  // namespace

// Converts audio buffers into tensors, possibly with resampling, buffering
// and framing, according to specified inputs and options. All input audio
// buffers will be first resampled from the input sample rate to the target
// sample rate if they are not equal. The resampled audio data (with the
// buffered samples from the previous runs in the streaming mode) will be broken
// into fixed-sized, possibly overlapping frames. If the calculator is not asked
// to perform fft (the fft_size is not set in the calculator options), all
// frames will be converted to and outputted as MediaPipe Tensors. The last
// output tensor will be zero-padding if the remaining samples are insufficient.
// Otherwise, when the fft_size is set and valid, the calculator will perform
// fft on the fixed-sized audio frames, the complex DFT results will be
// converted to and outputted as 2D MediaPipe float Tensors where the first
// rows are the DFT real parts and the second rows are the DFT imagery parts.
//
// This calculator assumes that the input timestamps refer to the first
// sample in each Matrix. The output timestamps follow this same convention.
// One Process() call may output multiple tensors packets. The timestamps of
// the output packets are determined by the timestamp of the previous output
// packet, the target sample rate, and the number of samples advanced after the
// previous output.
//
// The calculator has two running modes:
//   Streaming mode: when "stream_mode" is set to true in the calculator
//     options, the calculator treats the input audio stream as a continuous
//     stream. Thus, any samples that are not consumed in the previous runs will
//     be cached in a global sample buffer. The audio data resampled from the
//     current raw audio input will be appended to the global sample buffer.
//     The calculator will process the global sample buffer and output as many
//     tensors as possible.
//   Non-streaming mode: when "stream_mode" is set to false in the calculator
//     options, the calculators treats the packets in the input audio stream as
//     a batch of unrelated audio buffers. In each Process() call, the input
//     buffer will be first resampled, and framed as fixed-sized, possibly
//     overlapping tensors. The last tensor produced by a Process() invocation
//     will be zero-padding if the remaining samples are insufficient. As the
//     calculator treats the input packets as unrelated, all samples will be
//     processed immediately and no samples will be cached in the global sample
//     buffer.
//
// Inputs:
//   AUDIO - mediapipe::Matrix
//     The audio data represented as mediapipe::Matrix.
//   SAMPLE_RATE - double @Optional
//     The sample rate of the corresponding audio data in the "AUDIO" stream.
//     If a sample rate packet is provided at Timestamp::PreStream(), the sample
//     rate will be used as the sample rate of every audio packets in the
//     "AUDIO" stream. Note that one and only one of the "AUDIO" stream's time
//     series header or the "SAMPLE_RATE" stream can exist.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor that represents a fix-sized audio
//     frame or the complex DFT results.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     Vector containing the output timestamps emitted by the current Process()
//     invocation. In the non-streaming mode, the vector contains all of the
//     output timestamps for an input audio buffer.
//   DC_AND_NYQUIST - std::pair<float, float> @Optional.
//     A pair of dc component and nyquist component. Only can be connected when
//     the calculator performs fft (the fft_size is set in the calculator
//     options).
//
// Example:
// node {
//   calculator: "AudioToTensorCalculator"
//   input_stream: "AUDIO:audio"
//   output_stream: "TENSORS:tensors"
//   output_stream: "TIMESTAMPS:timestamps"
//   options {
//     [mediapipe.AudioToTensorCalculatorOptions.ext] {
//       num_channels: 2
//       num_samples: 512
//       num_overlapping_samples: 64
//       target_sample_rate: 16000
//       stream_mode: true # or false
//     }
//   }
// }
class AudioToTensorCalculator : public Node {
 public:
  static constexpr Input<Matrix> kAudioIn{"AUDIO"};
  // TODO: Removes this optional input stream when the "AUDIO" stream
  // uses the new mediapipe audio data containers that carry audio metadata,
  // such as sample rate.
  static constexpr Input<double>::Optional kAudioSampleRateIn{"SAMPLE_RATE"};
  static constexpr Output<std::vector<Tensor>> kTensorsOut{"TENSORS"};
  static constexpr Output<std::pair<float, float>>::Optional kDcAndNyquistOut{
      "DC_AND_NYQUIST"};
  // A vector of the output timestamps emitted by the current Process()
  // invocation. The packet timestamp is the last emitted timestamp.
  static constexpr Output<std::vector<Timestamp>>::Optional kTimestampsOut{
      "TIMESTAMPS"};
  MEDIAPIPE_NODE_CONTRACT(kAudioIn, kAudioSampleRateIn, kTensorsOut,
                          kDcAndNyquistOut, kTimestampsOut);

  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);
  absl::Status Close(CalculatorContext* cc);

 private:
  // The target number of channels.
  int num_channels_;
  // The target number of samples per channel.
  int num_samples_;
  // The number of samples per channel to advance after the current frame is
  // processed.
  int frame_step_;
  bool stream_mode_;
  bool check_inconsistent_timestamps_;
  int padding_samples_before_;
  int padding_samples_after_;
  FlushMode flush_mode_;
  DftTensorFormat dft_tensor_format_;

  Timestamp initial_timestamp_ = Timestamp::Unstarted();
  int64_t cumulative_input_samples_ = 0;
  Timestamp next_output_timestamp_ = Timestamp::Unstarted();

  double source_sample_rate_ = -1;
  double target_sample_rate_ = -1;
  // TODO: Configures QResamplerParams through calculator options.
  audio_dsp::QResamplerParams params_;
  // A QResampler instance to resample an audio stream.
  std::unique_ptr<audio_dsp::QResampler<float>> resampler_;
  Matrix sample_buffer_;
  int processed_buffer_cols_ = 0;
  double gain_ = 1.0;

  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;

  // The internal state of the FFT library.
  PFFFT_Setup* fft_state_ = nullptr;
  int fft_size_ = 0;
  std::vector<float> fft_window_;
  std::vector<float, Eigen::aligned_allocator<float>> fft_input_buffer_;
  // pffft requires memory to work with to avoid using the stack.
  std::vector<float, Eigen::aligned_allocator<float>> fft_workplace_;
  std::vector<float, Eigen::aligned_allocator<float>> fft_output_;

  absl::Status ProcessStreamingData(CalculatorContext* cc, const Matrix& input);
  absl::Status ProcessNonStreamingData(CalculatorContext* cc,
                                       const Matrix& input);

  absl::Status SetupStreamingResampler(double input_sample_rate_);
  void AppendToSampleBuffer(Matrix buffer_to_append);
  void AppendZerosToSampleBuffer(int num_samples);

  absl::StatusOr<std::vector<Tensor>> ConvertToTensor(
      const Matrix& block, std::vector<int> tensor_dims);
  absl::Status OutputTensor(const Matrix& block, Timestamp timestamp,
                            CalculatorContext* cc);
  absl::Status ProcessBuffer(const Matrix& buffer, bool should_flush,
                             CalculatorContext* cc);
};

absl::Status AudioToTensorCalculator::UpdateContract(CalculatorContract* cc) {
  const auto& options = cc->Options<Options>();
  if (!options.has_num_channels() || !options.has_num_samples() ||
      !options.has_target_sample_rate()) {
    return absl::InvalidArgumentError(
        "AudioToTensorCalculatorOptions must specify "
        "`num_channels`, `num_samples`, and `target_sample_rate`.");
  }
  if (options.stream_mode()) {
    // Explicitly disables timestamp offset to disallow the timestamp bound
    // from the input streams to be propagated to the output streams.
    // In the streaming mode, the output timestamp bound is based on
    // next_output_timestamp_, which can be smaller than the current input
    // timestamps.
    cc->SetTimestampOffset(TimestampDiff::Unset());
  }
  if (options.padding_samples_before() < 0 ||
      options.padding_samples_after() < 0) {
    return absl::InvalidArgumentError("Negative zero padding unsupported");
  }
  if (options.flush_mode() != Options::ENTIRE_TAIL_AT_TIMESTAMP_MAX &&
      options.flush_mode() != Options::PROCEED_AS_USUAL) {
    return absl::InvalidArgumentError("Unsupported flush mode");
  }
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

absl::Status AudioToTensorCalculator::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  const auto& options =
      cc->Options<mediapipe::AudioToTensorCalculatorOptions>();
  num_channels_ = options.num_channels();
  num_samples_ = options.num_samples();
  if (options.has_num_overlapping_samples()) {
    RET_CHECK_GE(options.num_overlapping_samples(), 0);
    RET_CHECK_LT(options.num_overlapping_samples(), num_samples_);
    frame_step_ = num_samples_ - options.num_overlapping_samples();
  } else {
    frame_step_ = num_samples_;
  }
  target_sample_rate_ = options.target_sample_rate();
  stream_mode_ = options.stream_mode();
  if (stream_mode_) {
    check_inconsistent_timestamps_ = options.check_inconsistent_timestamps();
    sample_buffer_.resize(num_channels_, Eigen::NoChange);
  }
  padding_samples_before_ = options.padding_samples_before();
  padding_samples_after_ = options.padding_samples_after();
  dft_tensor_format_ = options.dft_tensor_format();
  flush_mode_ = options.flush_mode();
  if (options.has_volume_gain_db()) {
    gain_ = pow(10, options.volume_gain_db() / 20.0);
  }
  if (options.has_source_sample_rate()) {
    source_sample_rate_ = options.source_sample_rate();
  } else {
    RET_CHECK(kAudioSampleRateIn(cc).IsConnected() ^
              !kAudioIn(cc).Header().IsEmpty())
        << "Must either specify the time series header of the \"AUDIO\" stream "
           "or have the \"SAMPLE_RATE\" stream connected.";
    if (!kAudioIn(cc).Header().IsEmpty()) {
      mediapipe::TimeSeriesHeader input_header;
      MP_RETURN_IF_ERROR(
          mediapipe::time_series_util::FillTimeSeriesHeaderIfValid(
              kAudioIn(cc).Header(), &input_header));
      if (stream_mode_) {
        MP_RETURN_IF_ERROR(SetupStreamingResampler(input_header.sample_rate()));
      } else {
        source_sample_rate_ = input_header.sample_rate();
      }
    }
  }
  AppendZerosToSampleBuffer(padding_samples_before_);
  if (options.has_fft_size()) {
    RET_CHECK(IsValidFftSize(options.fft_size()))
        << "FFT size must be of the form fft_size = (2^a)*(3^b)*(5^c) where b "
           ">=0 and c >= 0 and a >= 5, the requested fft size is "
        << options.fft_size();
    RET_CHECK_EQ(1, num_channels_)
        << "Currently only support applying FFT on mono channel.";
    fft_size_ = options.fft_size();
    fft_state_ = pffft_new_setup(fft_size_, PFFFT_REAL);
    fft_window_ = HannWindow(fft_size_, /* sqrt_hann = */ false);
    fft_input_buffer_.resize(fft_size_);
    fft_workplace_.resize(fft_size_);
    fft_output_.resize(fft_size_);
  } else {
    RET_CHECK(!kDcAndNyquistOut(cc).IsConnected())
        << "The DC_AND_NYQUIST output stream can only be connected when the "
           "calculator outputs fft tensors";
  }
  return absl::OkStatus();
}

absl::Status AudioToTensorCalculator::Process(CalculatorContext* cc) {
  if (cc->InputTimestamp() == Timestamp::PreStream()) {
    double current_source_sample_rate = kAudioSampleRateIn(cc).Get();
    if (cc->Options<mediapipe::AudioToTensorCalculatorOptions>()
            .stream_mode()) {
      return SetupStreamingResampler(current_source_sample_rate);
    } else {
      source_sample_rate_ = current_source_sample_rate;
      return absl::OkStatus();
    }
  }
  // Sanity checks.
  const auto& input_frame = kAudioIn(cc).Get();
  const bool channels_match = input_frame.rows() == num_channels_;
  // The special case of `num_channels_ == 1` is automatic mixdown to mono.
  const bool mono_output = num_channels_ == 1;
  if (!mono_output && !channels_match) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Audio input has %d channel(s) but the model requires %d channel(s).",
        input_frame.rows(), num_channels_));
  }
  if (!mono_output && input_frame.IsRowMajor) {
    return absl::InvalidArgumentError(
        "The audio data should be stored in column-major.");
  }
  ABSL_CHECK(channels_match || mono_output);
  const Matrix& input = channels_match ? input_frame
                                       // Mono mixdown.
                                       : input_frame.colwise().mean();
  if (gain_ != 1.0) {
    return stream_mode_ ? ProcessStreamingData(cc, input * gain_)
                        : ProcessNonStreamingData(cc, input * gain_);
  }
  return stream_mode_ ? ProcessStreamingData(cc, input)
                      : ProcessNonStreamingData(cc, input);
}

absl::Status AudioToTensorCalculator::Close(CalculatorContext* cc) {
  if (!stream_mode_) {
    return absl::OkStatus();
  }
  if (resampler_) {
    Matrix resampled_buffer(num_channels_, 0);
    resampler_->Flush(&resampled_buffer);
    AppendToSampleBuffer(std::move(resampled_buffer));
  }
  AppendZerosToSampleBuffer(padding_samples_after_);
  MP_RETURN_IF_ERROR(ProcessBuffer(sample_buffer_, /*should_flush=*/true, cc));
  if (fft_state_) {
    pffft_destroy_setup(fft_state_);
  }
  return absl::OkStatus();
}

absl::Status AudioToTensorCalculator::ProcessStreamingData(
    CalculatorContext* cc, const Matrix& input) {
  const auto& input_buffer = input;
  if (initial_timestamp_ == Timestamp::Unstarted()) {
    initial_timestamp_ = cc->InputTimestamp();
    next_output_timestamp_ = initial_timestamp_;
  }
  if (source_sample_rate_ != -1 && check_inconsistent_timestamps_) {
    mediapipe::time_series_util::LogWarningIfTimestampIsInconsistent(
        cc->InputTimestamp(), initial_timestamp_, cumulative_input_samples_,
        source_sample_rate_);
    cumulative_input_samples_ += input_buffer.cols();
  }
  if (!kAudioSampleRateIn(cc).IsEmpty()) {
    double current_source_sample_rate = kAudioSampleRateIn(cc).Get();
    if (resampler_) {
      RET_CHECK_EQ(current_source_sample_rate, source_sample_rate_);
    } else {
      MP_RETURN_IF_ERROR(SetupStreamingResampler(current_source_sample_rate));
    }
  }

  if (resampler_) {
    Matrix resampled_buffer(num_channels_, 0);
    resampler_->ProcessSamples(input_buffer, &resampled_buffer);
    AppendToSampleBuffer(std::move(resampled_buffer));
  } else {
    // Tries to consume the input matrix first to avoid extra data copy.
    auto status_or_matrix = kAudioIn(cc).packet().Consume<Matrix>();
    if (status_or_matrix.ok()) {
      Matrix local_matrix(num_channels_, 0);
      local_matrix.swap(*status_or_matrix.value());
      AppendToSampleBuffer(std::move(local_matrix));
    } else {
      AppendToSampleBuffer(input_buffer);
    }
  }

  MP_RETURN_IF_ERROR(ProcessBuffer(sample_buffer_, /*should_flush=*/false, cc));
  // Removes the processed samples from the global sample buffer.
  sample_buffer_ = Matrix(sample_buffer_.rightCols(sample_buffer_.cols() -
                                                   processed_buffer_cols_ - 1));
  return absl::OkStatus();
}

absl::Status AudioToTensorCalculator::ProcessNonStreamingData(
    CalculatorContext* cc, const Matrix& input) {
  initial_timestamp_ = cc->InputTimestamp();
  next_output_timestamp_ = initial_timestamp_;
  const auto& input_frame = input;
  double source_sample_rate = kAudioSampleRateIn(cc).GetOr(source_sample_rate_);

  if (source_sample_rate != -1 && source_sample_rate != target_sample_rate_) {
    std::vector<float> resampled = audio_dsp::QResampleSignal<float>(
        source_sample_rate, target_sample_rate_, num_channels_, params_,
        input_frame);
    Eigen::Map<const Matrix> matrix_mapping(resampled.data(), num_channels_,
                                            resampled.size() / num_channels_);
    return ProcessBuffer(matrix_mapping, /*should_flush=*/true, cc);
  }
  return ProcessBuffer(input_frame, /*should_flush=*/true, cc);
}

absl::Status AudioToTensorCalculator::SetupStreamingResampler(
    double input_sample_rate) {
  if (input_sample_rate == source_sample_rate_) {
    return absl::OkStatus();
  }
  source_sample_rate_ = input_sample_rate;
  if (source_sample_rate_ != target_sample_rate_) {
    resampler_ = absl::make_unique<audio_dsp::QResampler<float>>(
        source_sample_rate_, target_sample_rate_, num_channels_, params_);
    if (!resampler_) {
      return absl::InternalError("Failed to initialize resampler.");
    }
  }
  return absl::OkStatus();
}

void AudioToTensorCalculator::AppendZerosToSampleBuffer(int num_samples) {
  ABSL_CHECK_GE(num_samples, 0);  // Ensured by `UpdateContract`.
  if (num_samples == 0) {
    return;
  }
  sample_buffer_.conservativeResize(Eigen::NoChange,
                                    sample_buffer_.cols() + num_samples);
  sample_buffer_.rightCols(num_samples).setZero();
}

void AudioToTensorCalculator::AppendToSampleBuffer(Matrix buffer_to_append) {
  sample_buffer_.conservativeResize(
      Eigen::NoChange, sample_buffer_.cols() + buffer_to_append.cols());
  sample_buffer_.rightCols(buffer_to_append.cols()).swap(buffer_to_append);
}

absl::StatusOr<std::vector<Tensor>> AudioToTensorCalculator::ConvertToTensor(
    const Matrix& block, std::vector<int> tensor_dims) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape(tensor_dims),
                memory_manager_);
  auto buffer_view = tensor.GetCpuWriteView();
  int total_size = 1;
  for (int dim : tensor_dims) {
    total_size *= dim;
  }
  if (block.size() < total_size) {
    std::memset(buffer_view.buffer<float>(), 0, tensor.bytes());
  }
  std::memcpy(buffer_view.buffer<float>(), block.data(),
              block.size() * sizeof(float));
  std::vector<Tensor> tensor_vector;
  tensor_vector.push_back(std::move(tensor));
  return tensor_vector;
}

absl::Status AudioToTensorCalculator::OutputTensor(const Matrix& block,
                                                   Timestamp timestamp,
                                                   CalculatorContext* cc) {
  std::vector<Tensor> output_tensor;
  if (fft_state_) {
    Eigen::VectorXf time_series_data =
        Eigen::VectorXf::Map(block.data(), block.size());
    //  Window on input audio prior to FFT.
    std::transform(time_series_data.begin(), time_series_data.end(),
                   fft_window_.begin(), fft_input_buffer_.begin(),
                   std::multiplies<float>());
    pffft_transform_ordered(fft_state_, fft_input_buffer_.data(),
                            fft_output_.data(), fft_workplace_.data(),
                            PFFFT_FORWARD);
    if (kDcAndNyquistOut(cc).IsConnected()) {
      kDcAndNyquistOut(cc).Send(std::make_pair(fft_output_[0], fft_output_[1]),
                                timestamp);
    }
    switch (dft_tensor_format_) {
      case Options::WITH_NYQUIST: {
        Matrix fft_output_matrix =
            Eigen::Map<const Matrix>(fft_output_.data() + 2, 1, fft_size_ - 2);
        fft_output_matrix.conservativeResize(Eigen::NoChange, fft_size_);
        // The last two elements are Nyquist component.
        fft_output_matrix(fft_size_ - 2) = fft_output_[1];  // Nyquist real part
        fft_output_matrix(fft_size_ - 1) = 0.0f;  // Nyquist imagery part
        MP_ASSIGN_OR_RETURN(output_tensor, ConvertToTensor(fft_output_matrix,
                                                           {2, fft_size_ / 2}));
        break;
      }
      case Options::WITH_DC_AND_NYQUIST: {
        Matrix fft_output_matrix =
            Eigen::Map<const Matrix>(fft_output_.data(), 1, fft_size_);
        fft_output_matrix.conservativeResize(Eigen::NoChange, fft_size_ + 2);
        fft_output_matrix(1) = 0.0f;  // DC imagery part.
        // The last two elements are  Nyquist component.
        fft_output_matrix(fft_size_) = fft_output_[1];  // Nyquist real part
        fft_output_matrix(fft_size_ + 1) = 0.0f;        // Nyquist imagery part
        MP_ASSIGN_OR_RETURN(
            output_tensor,
            ConvertToTensor(fft_output_matrix, {2, (fft_size_ + 2) / 2}));
        break;
      }
      case Options::WITHOUT_DC_AND_NYQUIST: {
        Matrix fft_output_matrix =
            Eigen::Map<const Matrix>(fft_output_.data() + 2, 1, fft_size_ - 2);
        MP_ASSIGN_OR_RETURN(
            output_tensor,
            ConvertToTensor(fft_output_matrix, {2, (fft_size_ - 2) / 2}));
        break;
      }
      default:
        return absl::InvalidArgumentError("Unsupported dft tensor format.");
    }

  } else {
    MP_ASSIGN_OR_RETURN(output_tensor,
                        ConvertToTensor(block, {num_channels_, num_samples_}));
  }
  kTensorsOut(cc).Send(std::move(output_tensor), timestamp);
  return absl::OkStatus();
}

absl::Status AudioToTensorCalculator::ProcessBuffer(const Matrix& buffer,
                                                    bool should_flush,
                                                    CalculatorContext* cc) {
  const bool should_flush_at_timestamp_max =
      stream_mode_ && should_flush &&
      flush_mode_ == Options::ENTIRE_TAIL_AT_TIMESTAMP_MAX;
  int next_frame_first_col = 0;
  std::vector<Timestamp> timestamps;
  if (!should_flush_at_timestamp_max) {
    while (next_frame_first_col + num_samples_ <= buffer.cols()) {
      MP_RETURN_IF_ERROR(OutputTensor(
          buffer.block(0, next_frame_first_col, num_channels_, num_samples_),
          next_output_timestamp_, cc));
      timestamps.push_back(next_output_timestamp_);
      next_output_timestamp_ += round(frame_step_ / target_sample_rate_ *
                                      Timestamp::kTimestampUnitsPerSecond);
      next_frame_first_col += frame_step_;
    }
  }
  if (should_flush && next_frame_first_col < buffer.cols()) {
    // In the streaming mode, the flush happens in Close() and a packet at
    // Timestamp::Max() will be emitted. In the non-streaming mode, each
    // Process() invocation will process the entire buffer completely.
    Timestamp timestamp = should_flush_at_timestamp_max
                              ? Timestamp::Max()
                              : next_output_timestamp_;
    MP_RETURN_IF_ERROR(OutputTensor(
        buffer.block(
            0, next_frame_first_col, num_channels_,
            std::min(num_samples_, (int)buffer.cols() - next_frame_first_col)),
        timestamp, cc));
    timestamps.push_back(timestamp);
  }
  if (kTimestampsOut(cc).IsConnected()) {
    Timestamp timestamp = timestamps.back();
    kTimestampsOut(cc).Send(std::move(timestamps), timestamp);
  }
  processed_buffer_cols_ = next_frame_first_col - 1;
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(AudioToTensorCalculator);

}  // namespace api2
}  // namespace mediapipe
