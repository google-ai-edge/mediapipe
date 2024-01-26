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

#include <cmath>
#include <cstdint>

#include "Eigen/Core"
#include "mediapipe/calculators/audio/stabilized_log_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {

const float kStabilizer = 0.1;
const int kNumChannels = 3;
const int kNumSamples = 10;

class StabilizedLogCalculatorTest
    : public TimeSeriesCalculatorTest<StabilizedLogCalculatorOptions> {
 protected:
  void SetUp() override {
    calculator_name_ = "StabilizedLogCalculator";
    options_.set_stabilizer(kStabilizer);

    input_sample_rate_ = 8000.0;
    num_input_channels_ = kNumChannels;
    num_input_samples_ = kNumSamples;
  }

  void RunGraphNoReturn() { MP_ASSERT_OK(RunGraph()); }
};

TEST_F(StabilizedLogCalculatorTest, BasicOperation) {
  const int kNumPackets = 5;

  InitializeGraph();
  FillInputHeader();

  std::vector<Matrix> input_data_matrices;
  for (int input_packet = 0; input_packet < kNumPackets; ++input_packet) {
    const int64_t timestamp =
        input_packet * Timestamp::kTimestampUnitsPerSecond;
    Matrix input_data_matrix =
        Matrix::Random(kNumChannels, kNumSamples).array().abs();
    input_data_matrices.push_back(input_data_matrix);
    AppendInputPacket(new Matrix(input_data_matrix), timestamp);
  }

  MP_ASSERT_OK(RunGraph());
  ExpectOutputHeaderEqualsInputHeader();
  for (int output_packet = 0; output_packet < kNumPackets; ++output_packet) {
    ExpectApproximatelyEqual(
        (input_data_matrices[output_packet].array() + kStabilizer).log(),
        runner_->Outputs().Index(0).packets[output_packet].Get<Matrix>());
  }
}

TEST_F(StabilizedLogCalculatorTest, OutputScaleWorks) {
  const int kNumPackets = 5;
  double output_scale = 2.5;
  options_.set_output_scale(output_scale);

  InitializeGraph();
  FillInputHeader();

  std::vector<Matrix> input_data_matrices;
  for (int input_packet = 0; input_packet < kNumPackets; ++input_packet) {
    const int64_t timestamp =
        input_packet * Timestamp::kTimestampUnitsPerSecond;
    Matrix input_data_matrix =
        Matrix::Random(kNumChannels, kNumSamples).array().abs();
    input_data_matrices.push_back(input_data_matrix);
    AppendInputPacket(new Matrix(input_data_matrix), timestamp);
  }

  MP_ASSERT_OK(RunGraph());
  ExpectOutputHeaderEqualsInputHeader();
  for (int output_packet = 0; output_packet < kNumPackets; ++output_packet) {
    ExpectApproximatelyEqual(
        output_scale *
            ((input_data_matrices[output_packet].array() + kStabilizer).log()),
        runner_->Outputs().Index(0).packets[output_packet].Get<Matrix>());
  }
}

TEST_F(StabilizedLogCalculatorTest, ZerosAreStabilized) {
  InitializeGraph();
  FillInputHeader();
  AppendInputPacket(new Matrix(Matrix::Zero(kNumChannels, kNumSamples)),
                    0 /* timestamp */);
  MP_ASSERT_OK(RunGraph());
  ExpectOutputHeaderEqualsInputHeader();
  ExpectApproximatelyEqual(
      Matrix::Constant(kNumChannels, kNumSamples, kStabilizer).array().log(),
      runner_->Outputs().Index(0).packets[0].Get<Matrix>());
}

TEST_F(StabilizedLogCalculatorTest, NanValuesReturnError) {
  InitializeGraph();
  FillInputHeader();
  AppendInputPacket(
      new Matrix(Matrix::Constant(kNumChannels, kNumSamples, std::nanf(""))),
      0 /* timestamp */);
  ASSERT_FALSE(RunGraph().ok());
}

TEST_F(StabilizedLogCalculatorTest, NegativeValuesReturnError) {
  InitializeGraph();
  FillInputHeader();
  AppendInputPacket(
      new Matrix(Matrix::Constant(kNumChannels, kNumSamples, -1.0)),
      0 /* timestamp */);
  ASSERT_FALSE(RunGraph().ok());
}

TEST_F(StabilizedLogCalculatorTest, NegativeValuesDoNotCheckFailIfCheckIsOff) {
  options_.set_check_nonnegativity(false);
  InitializeGraph();
  FillInputHeader();
  AppendInputPacket(
      new Matrix(Matrix::Constant(kNumChannels, kNumSamples, -1.0)),
      0 /* timestamp */);
  MP_ASSERT_OK(RunGraph());
  // Results are undefined.
}

}  // namespace mediapipe
