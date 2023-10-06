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
#include <new>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "audio/dsp/window_functions.h"
#include "mediapipe/calculators/tensor/tensors_to_audio_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "pffft.h"

namespace mediapipe {
namespace api2 {
namespace {

using Options = ::mediapipe::TensorsToAudioCalculatorOptions;

std::vector<float> HannWindow(int window_size, bool sqrt_hann) {
  std::vector<float> hann_window(window_size);
  audio_dsp::HannWindow().GetPeriodicSamples(window_size, &hann_window);
  if (sqrt_hann) {
    absl::c_transform(hann_window, hann_window.begin(),
                      [](double x) { return std::sqrt(x); });
  }
  return hann_window;
}

// Note that the InvHannWindow function may only work for 50% overlapping case.
std::vector<float> InvHannWindow(int window_size, bool sqrt_hann) {
  std::vector<float> window = HannWindow(window_size, sqrt_hann);
  std::vector<float> inv_window(window.size());
  if (sqrt_hann) {
    absl::c_copy(window, inv_window.begin());
  } else {
    const int kHalfWindowSize = window.size() / 2;
    absl::c_transform(window, inv_window.begin(),
                      [](double x) { return x * x; });
    for (int i = 0; i < kHalfWindowSize; ++i) {
      double sum = inv_window[i] + inv_window[kHalfWindowSize + i];
      inv_window[i] = window[i] / sum;
      inv_window[kHalfWindowSize + i] = window[kHalfWindowSize + i] / sum;
    }
  }
  return inv_window;
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

// Converts 2D MediaPipe float Tensors to audio buffers.
// The calculator will perform ifft on the complex DFT and apply the window
// function (Inverse Hann) afterwards. The input 2D MediaPipe Tensor must
// have the DFT real parts in its first row and the DFT imagery parts in its
// second row. A valid "fft_size" must be set in the CalculatorOptions.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor that represents the audio's complex DFT
//     results.
//   DC_AND_NYQUIST - std::pair<float, float>
//     A pair of dc component and nyquist component.
//
// Outputs:
//   AUDIO - mediapipe::Matrix
//     The audio data represented as mediapipe::Matrix.
//
// Example:
// node {
//   calculator: "TensorsToAudioCalculator"
//   input_stream: "TENSORS:tensors"
//   input_stream: "DC_AND_NYQUIST:dc_and_nyquist"
//   output_stream: "AUDIO:audio"
//   options {
//     [mediapipe.AudioToTensorCalculatorOptions.ext] {
//       fft_size: 256
//     }
//   }
// }
class TensorsToAudioCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kTensorsIn{"TENSORS"};
  static constexpr Input<std::pair<float, float>> kDcAndNyquistIn{
      "DC_AND_NYQUIST"};
  static constexpr Output<Matrix> kAudioOut{"AUDIO"};
  MEDIAPIPE_NODE_CONTRACT(kTensorsIn, kDcAndNyquistIn, kAudioOut);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  // The internal state of the FFT library.
  PFFFT_Setup* fft_state_ = nullptr;
  int fft_size_ = 0;
  float inverse_fft_size_ = 0;
  std::vector<float, Eigen::aligned_allocator<float>> input_dft_;
  std::vector<float> inv_fft_window_;
  std::vector<float, Eigen::aligned_allocator<float>> fft_input_buffer_;
  // pffft requires memory to work with to avoid using the stack.
  std::vector<float, Eigen::aligned_allocator<float>> fft_workplace_;
  std::vector<float, Eigen::aligned_allocator<float>> fft_output_;
  std::vector<float, Eigen::aligned_allocator<float>> prev_fft_output_;
  int overlapping_samples_ = -1;
  int step_samples_ = -1;
  Options::DftTensorFormat dft_tensor_format_;
  double gain_ = 1.0;
};

absl::Status TensorsToAudioCalculator::Open(CalculatorContext* cc) {
  const auto& options =
      cc->Options<mediapipe::TensorsToAudioCalculatorOptions>();
  dft_tensor_format_ = options.dft_tensor_format();
  RET_CHECK(dft_tensor_format_ != Options::DFT_TENSOR_FORMAT_UNKNOWN)
      << "dft tensor format must be specified.";
  RET_CHECK(options.has_fft_size()) << "FFT size must be specified.";
  RET_CHECK(IsValidFftSize(options.fft_size()))
      << "FFT size must be of the form fft_size = (2^a)*(3^b)*(5^c) where b "
         ">=0 and c >= 0 and a >= 5, the requested fft size is "
      << options.fft_size();
  fft_size_ = options.fft_size();
  inverse_fft_size_ = 1.0f / fft_size_;
  fft_state_ = pffft_new_setup(fft_size_, PFFFT_REAL);
  input_dft_.resize(fft_size_);
  inv_fft_window_ = InvHannWindow(fft_size_, /* sqrt_hann = */ false);
  fft_input_buffer_.resize(fft_size_);
  fft_workplace_.resize(fft_size_);
  fft_output_.resize(fft_size_);
  if (options.has_num_overlapping_samples()) {
    RET_CHECK(options.has_num_samples() && options.num_samples() > 0)
        << "When `num_overlapping_samples` is set, `num_samples` must also be "
           "specified.";
    if (options.num_samples() != fft_size_) {
      return absl::UnimplementedError(
          "`num_samples` and `fft_size` must be equivalent.");
    }
    RET_CHECK(options.num_overlapping_samples() > 0 &&
              options.num_overlapping_samples() < options.num_samples())
        << "`num_overlapping_samples` must be greater than 0 and less than "
           "`num_samples.`";
    overlapping_samples_ = options.num_overlapping_samples();
    step_samples_ = options.num_samples() - options.num_overlapping_samples();
    prev_fft_output_.resize(fft_size_);
  }
  if (options.has_volume_gain_db()) {
    gain_ = pow(10, options.volume_gain_db() / 20.0);
  }
  return absl::OkStatus();
}

absl::Status TensorsToAudioCalculator::Process(CalculatorContext* cc) {
  if (kTensorsIn(cc).IsEmpty() || kDcAndNyquistIn(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_tensors = *kTensorsIn(cc);
  RET_CHECK_EQ(input_tensors.size(), 1);
  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  auto view = input_tensors[0].GetCpuReadView();
  switch (dft_tensor_format_) {
    case Options::WITH_NYQUIST: {
      // DC's real part.
      input_dft_[0] = kDcAndNyquistIn(cc)->first;
      // Nyquist's real part is the penultimate element of the tensor buffer.
      // pffft ignores the Nyquist's imagery part. No need to fetch the last
      // value from the tensor buffer.
      input_dft_[1] = *(view.buffer<float>() + (fft_size_ - 2));
      std::memcpy(input_dft_.data() + 2, view.buffer<float>(),
                  (fft_size_ - 2) * sizeof(float));
      break;
    }
    case Options::WITH_DC_AND_NYQUIST: {
      // DC's real part is the first element of the tensor buffer.
      input_dft_[0] = *(view.buffer<float>());
      // Nyquist's real part is the penultimate element of the tensor buffer.
      input_dft_[1] = *(view.buffer<float>() + fft_size_);
      std::memcpy(input_dft_.data() + 2, view.buffer<float>() + 2,
                  (fft_size_ - 2) * sizeof(float));
      break;
    }
    case Options::WITHOUT_DC_AND_NYQUIST: {
      input_dft_[0] = kDcAndNyquistIn(cc)->first;
      input_dft_[1] = kDcAndNyquistIn(cc)->second;
      std::memcpy(input_dft_.data() + 2, view.buffer<float>(),
                  (fft_size_ - 2) * sizeof(float));
      break;
    }
    default:
      return absl::InvalidArgumentError("Unsupported dft tensor format.");
  }
  pffft_transform_ordered(fft_state_, input_dft_.data(), fft_output_.data(),
                          fft_workplace_.data(), PFFFT_BACKWARD);
  // Applies the inverse window function.
  std::transform(
      fft_output_.begin(), fft_output_.end(), inv_fft_window_.begin(),
      fft_output_.begin(),
      [this](float a, float b) { return a * b * inverse_fft_size_; });
  Matrix matrix;
  if (step_samples_ > 0) {
    matrix = Eigen::Map<Matrix>(fft_output_.data(), 1, step_samples_);
    matrix.leftCols(overlapping_samples_) += Eigen::Map<Matrix>(
        prev_fft_output_.data() + step_samples_, 1, overlapping_samples_);
    prev_fft_output_.swap(fft_output_);
  } else {
    matrix = Eigen::Map<Matrix>(fft_output_.data(), 1, fft_output_.size());
  }
  if (gain_ != 1.0) {
    matrix *= gain_;
  }
  kAudioOut(cc).Send(std::move(matrix));
  return absl::OkStatus();
}

absl::Status TensorsToAudioCalculator::Close(CalculatorContext* cc) {
  if (fft_state_) {
    pffft_destroy_setup(fft_state_);
  }
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(TensorsToAudioCalculator);

}  // namespace api2
}  // namespace mediapipe
