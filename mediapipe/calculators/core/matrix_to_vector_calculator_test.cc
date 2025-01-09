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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "mediapipe/util/time_series_test_util.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {
namespace {

class MatrixToVectorCalculatorTest
    : public mediapipe::TimeSeriesCalculatorTest<mediapipe::NoOptions> {
 protected:
  void SetUp() override { calculator_name_ = "MatrixToVectorCalculator"; }

  void AppendInput(const std::vector<float>& column_major_data,
                   int64_t timestamp) {
    ASSERT_EQ(num_input_samples_ * num_input_channels_,
              column_major_data.size());
    Eigen::Map<const Matrix> data_map(&column_major_data[0],
                                      num_input_channels_, num_input_samples_);
    AppendInputPacket(new Matrix(data_map), timestamp);
  }

  void SetInputStreamParameters(int num_channels, int num_samples) {
    num_input_channels_ = num_channels;
    num_input_samples_ = num_samples;
    input_sample_rate_ = 100;
    input_packet_rate_ = 20.0;
  }

  void SetInputHeader(int num_channels, int num_samples) {
    SetInputStreamParameters(num_channels, num_samples);
    FillInputHeader();
  }

  void CheckOutputPacket(int packet, std::vector<float> expected_vector) {
    const auto& actual_vector =
        runner_->Outputs().Index(0).packets[packet].Get<std::vector<float>>();
    EXPECT_THAT(actual_vector, testing::ContainerEq(expected_vector));
  }
};

TEST_F(MatrixToVectorCalculatorTest, SingleRow) {
  InitializeGraph();
  SetInputHeader(1, 4);  // 1 channel x 4 samples
  const std::vector<float>& data_vector = {1.0, 2.0, 3.0, 4.0};
  AppendInput(data_vector, 0);
  MP_ASSERT_OK(RunGraph());
  CheckOutputPacket(0, data_vector);
}

TEST_F(MatrixToVectorCalculatorTest, RegularMatrix) {
  InitializeGraph();
  SetInputHeader(4, 2);  // 4 channels x 2 samples
  // Actual data matrix is the transpose of the appearance below.
  const std::vector<float>& data_vector = {1.0, 2.0, 3.0, 4.0,
                                           5.0, 6.0, 7.0, 8.0};
  AppendInput(data_vector, 0);

  MP_ASSERT_OK(RunGraph());
  CheckOutputPacket(0, data_vector);
}

}  // namespace

}  // namespace mediapipe
