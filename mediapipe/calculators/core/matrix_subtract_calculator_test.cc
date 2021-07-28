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

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {
namespace {

constexpr char kMinuendTag[] = "MINUEND";
constexpr char kSubtrahendTag[] = "SUBTRAHEND";

// A 3x4 Matrix of random integers in [0,1000).
const char kMatrixText[] =
    "rows: 3\n"
    "cols: 4\n"
    "packed_data: 387\n"
    "packed_data: 940\n"
    "packed_data: 815\n"
    "packed_data: 825\n"
    "packed_data: 997\n"
    "packed_data: 884\n"
    "packed_data: 419\n"
    "packed_data: 763\n"
    "packed_data: 123\n"
    "packed_data:  30\n"
    "packed_data: 825\n"
    "packed_data: 299\n";

const char kMatrixText2[] =
    "rows: 3\n"
    "cols: 4\n"
    "packed_data: 388\n"
    "packed_data: 941\n"
    "packed_data: 816\n"
    "packed_data: 826\n"
    "packed_data: 998\n"
    "packed_data: 885\n"
    "packed_data: 420\n"
    "packed_data: 764\n"
    "packed_data: 124\n"
    "packed_data:  31\n"
    "packed_data: 826\n"
    "packed_data: 300\n";

TEST(MatrixSubtractCalculatorTest, WrongConfig) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MatrixSubtractCalculator"
        input_stream: "input_matrix"
        input_side_packet: "SUBTRAHEND:side_matrix"
        input_side_packet: "MINUEND:side_matrix2"
        output_stream: "output_matrix"
      )pb");
  CalculatorRunner runner(node_config);
  auto status = runner.Run();
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr(
          "only accepts exactly one input stream and one input side packet"));
}

TEST(MatrixSubtractCalculatorTest, WrongConfig2) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MatrixSubtractCalculator"
        input_side_packet: "SUBTRAHEND:side_matrix"
        input_stream: "SUBTRAHEND:side_matrix2"
        output_stream: "output_matrix"
      )pb");
  CalculatorRunner runner(node_config);
  auto status = runner.Run();
  EXPECT_THAT(status.message(), testing::HasSubstr("must be connected"));
  EXPECT_THAT(status.message(), testing::HasSubstr("not both"));
}

TEST(MatrixSubtractCalculatorTest, SubtractFromInput) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MatrixSubtractCalculator"
        input_stream: "MINUEND:input_matrix"
        input_side_packet: "SUBTRAHEND:side_matrix"
        output_stream: "output_matrix"
      )pb");
  CalculatorRunner runner(node_config);
  Matrix* side_matrix = new Matrix();
  MatrixFromTextProto(kMatrixText, side_matrix);
  runner.MutableSidePackets()->Tag(kSubtrahendTag) = Adopt(side_matrix);

  Matrix* input_matrix = new Matrix();
  MatrixFromTextProto(kMatrixText2, input_matrix);
  runner.MutableInputs()
      ->Tag(kMinuendTag)
      .packets.push_back(Adopt(input_matrix).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  EXPECT_EQ(1, runner.Outputs().Index(0).packets.size());

  EXPECT_EQ(Timestamp(0), runner.Outputs().Index(0).packets[0].Timestamp());
  const Eigen::MatrixXf& result =
      runner.Outputs().Index(0).packets[0].Get<Matrix>();
  ASSERT_EQ(3, result.rows());
  ASSERT_EQ(4, result.cols());
  EXPECT_NEAR(result.sum(), 12, 1e-5);
}

TEST(MatrixSubtractCalculatorTest, SubtractFromSideMatrix) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MatrixSubtractCalculator"
        input_stream: "SUBTRAHEND:input_matrix"
        input_side_packet: "MINUEND:side_matrix"
        output_stream: "output_matrix"
      )pb");
  CalculatorRunner runner(node_config);
  Matrix* side_matrix = new Matrix();
  MatrixFromTextProto(kMatrixText, side_matrix);
  runner.MutableSidePackets()->Tag(kMinuendTag) = Adopt(side_matrix);

  Matrix* input_matrix = new Matrix();
  MatrixFromTextProto(kMatrixText2, input_matrix);
  runner.MutableInputs()
      ->Tag(kSubtrahendTag)
      .packets.push_back(Adopt(input_matrix).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  EXPECT_EQ(1, runner.Outputs().Index(0).packets.size());

  EXPECT_EQ(Timestamp(0), runner.Outputs().Index(0).packets[0].Timestamp());
  const Eigen::MatrixXf& result =
      runner.Outputs().Index(0).packets[0].Get<Matrix>();
  ASSERT_EQ(3, result.rows());
  ASSERT_EQ(4, result.cols());
  EXPECT_NEAR(result.sum(), -12, 1e-5);
}

}  // namespace
}  // namespace mediapipe
