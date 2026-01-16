// Copyright 2024 The MediaPipe Authors.
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
// Defines TwoTapFirFilterCalculator.

#include <memory>
#include <utility>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "audio/linear_filters/two_tap_fir_filter.h"
#include "mediapipe/calculators/audio/stabilized_log_calculator.pb.h"
#include "mediapipe/calculators/audio/two_tap_fir_filter_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

using ::linear_filters::TwoTapFirFilter;
using ::mediapipe::Matrix;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;

// Wraps TwoTapFirFilter class to provide a two tap FIR filter:
// y[n] = gain_now* x[n] + gain_prev * x[n-1]
// It keeps the state of the filter over multiple calls to Process().
//
// Example:
// It can be used to implement a timedomain preemphasis filter
// y[n] = 1.0* x[n] + preemph * x[n-1]
// where gain_now is 1.0 and gain_prev is the preemph value (for HTK it's -0.97)
//
// node {
//   calculator: "TwoTapFirFilterCalculator"
//   input_stream: "INPUT:input"
//   output_stream: "OUTPUT:output"
//   node_options {
//     [type.googleapis.com/mediapipe.TwoTapFirFilterCalculatorOptions] {
//       gain_now: 1.0
//       gain_prev: -0.97  # preemph coefficient
//     }
//   }
// }

class TwoTapFirFilterCalculator : public Node {
 public:
  static constexpr char kInputTag[] = "INPUT";
  static constexpr char kOutputTag[] = "OUTPUT";

  static constexpr Input<Matrix> kInputSignal{kInputTag};
  static constexpr Output<Matrix> kOutputSignal{kOutputTag};

  MEDIAPIPE_NODE_CONTRACT(kInputSignal, kOutputSignal);

  absl::Status Open(CalculatorContext* cc) override {
    auto audio_header = std::make_unique<mediapipe::TimeSeriesHeader>();
    MP_RETURN_IF_ERROR(mediapipe::time_series_util::FillTimeSeriesHeaderIfValid(
        cc->Inputs().Tag(kInputTag).Header(), audio_header.get()));
    const auto& options = cc->Options<TwoTapFirFilterCalculatorOptions>();
    two_tap_fir_filter_ = std::make_unique<TwoTapFirFilter>(
        std::make_pair(options.gain_prev(), options.gain_now()));

    two_tap_fir_filter_->Init(audio_header->num_channels());

    // Output audio will have the same format as the original input.
    cc->Outputs().Tag(kOutputTag).SetHeader(Adopt(audio_header.release()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto input_matrix = kInputSignal(cc).Get();

    Eigen::ArrayXXf output(input_matrix.rows(), input_matrix.cols());
    two_tap_fir_filter_->ProcessBlock(input_matrix, &output);

    kOutputSignal(cc).Send(output.matrix(), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<TwoTapFirFilter> two_tap_fir_filter_;
};

REGISTER_CALCULATOR(TwoTapFirFilterCalculator);

}  // namespace mediapipe
