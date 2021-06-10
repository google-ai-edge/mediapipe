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
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {
namespace {

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

// A 4x20 Matrix of random integers in [0,10).
// Each column of this matrix is a sample.
const char kSamplesText[] =
    "rows: 4\n"
    "cols: 20\n"
    "packed_data: 7\n"
    "packed_data: 9\n"
    "packed_data: 5\n"
    "packed_data: 9\n"
    "packed_data: 6\n"
    "packed_data: 3\n"
    "packed_data: 0\n"
    "packed_data: 7\n"
    "packed_data: 1\n"
    "packed_data: 3\n"
    "packed_data: 3\n"
    "packed_data: 2\n"
    "packed_data: 4\n"
    "packed_data: 5\n"
    "packed_data: 0\n"
    "packed_data: 4\n"
    "packed_data: 6\n"
    "packed_data: 0\n"
    "packed_data: 1\n"
    "packed_data: 2\n"
    "packed_data: 0\n"
    "packed_data: 2\n"
    "packed_data: 0\n"
    "packed_data: 3\n"
    "packed_data: 1\n"
    "packed_data: 7\n"
    "packed_data: 4\n"
    "packed_data: 9\n"
    "packed_data: 8\n"
    "packed_data: 8\n"
    "packed_data: 6\n"
    "packed_data: 4\n"
    "packed_data: 6\n"
    "packed_data: 8\n"
    "packed_data: 1\n"
    "packed_data: 9\n"
    "packed_data: 7\n"
    "packed_data: 5\n"
    "packed_data: 3\n"
    "packed_data: 5\n"
    "packed_data: 3\n"
    "packed_data: 5\n"
    "packed_data: 7\n"
    "packed_data: 7\n"
    "packed_data: 3\n"
    "packed_data: 3\n"
    "packed_data: 6\n"
    "packed_data: 4\n"
    "packed_data: 7\n"
    "packed_data: 7\n"
    "packed_data: 2\n"
    "packed_data: 5\n"
    "packed_data: 4\n"
    "packed_data: 8\n"
    "packed_data: 1\n"
    "packed_data: 0\n"
    "packed_data: 2\n"
    "packed_data: 0\n"
    "packed_data: 3\n"
    "packed_data: 4\n"
    "packed_data: 6\n"
    "packed_data: 6\n"
    "packed_data: 8\n"
    "packed_data: 5\n"
    "packed_data: 5\n"
    "packed_data: 8\n"
    "packed_data: 9\n"
    "packed_data: 7\n"
    "packed_data: 3\n"
    "packed_data: 7\n"
    "packed_data: 2\n"
    "packed_data: 7\n"
    "packed_data: 8\n"
    "packed_data: 2\n"
    "packed_data: 1\n"
    "packed_data: 1\n"
    "packed_data: 4\n"
    "packed_data: 1\n"
    "packed_data: 1\n"
    "packed_data: 7\n";

// A 3x20 Matrix of expected values for the result of the matrix multiply
// computed using R.
// Each column of this matrix is an expected output.
const char kExpectedText[] =
    "rows: 3\n"
    "cols: 20\n"
    "packed_data: 12499\n"
    "packed_data: 26793\n"
    "packed_data: 16967\n"
    "packed_data:  5007\n"
    "packed_data: 14406\n"
    "packed_data:  9635\n"
    "packed_data:  4179\n"
    "packed_data:  7870\n"
    "packed_data:  4434\n"
    "packed_data:  5793\n"
    "packed_data: 12045\n"
    "packed_data:  8876\n"
    "packed_data:  2801\n"
    "packed_data:  8053\n"
    "packed_data:  5611\n"
    "packed_data:  1740\n"
    "packed_data:  4469\n"
    "packed_data:  2665\n"
    "packed_data:  8108\n"
    "packed_data: 18396\n"
    "packed_data: 10186\n"
    "packed_data: 12330\n"
    "packed_data: 23374\n"
    "packed_data: 15526\n"
    "packed_data:  9611\n"
    "packed_data: 21804\n"
    "packed_data: 14776\n"
    "packed_data:  8241\n"
    "packed_data: 17979\n"
    "packed_data: 11989\n"
    "packed_data:  8429\n"
    "packed_data: 18921\n"
    "packed_data:  9819\n"
    "packed_data:  6270\n"
    "packed_data: 13689\n"
    "packed_data:  7031\n"
    "packed_data:  9472\n"
    "packed_data: 19210\n"
    "packed_data: 13634\n"
    "packed_data:  8567\n"
    "packed_data: 12499\n"
    "packed_data: 10455\n"
    "packed_data:  2151\n"
    "packed_data:  7469\n"
    "packed_data:  3195\n"
    "packed_data: 10774\n"
    "packed_data: 21851\n"
    "packed_data: 12673\n"
    "packed_data: 12516\n"
    "packed_data: 25318\n"
    "packed_data: 14347\n"
    "packed_data:  7984\n"
    "packed_data: 17100\n"
    "packed_data: 10972\n"
    "packed_data:  5195\n"
    "packed_data: 11102\n"
    "packed_data:  8710\n"
    "packed_data:  3002\n"
    "packed_data: 11295\n"
    "packed_data:  6360\n";

// Send a number of samples through the MatrixMultiplyCalculator.
TEST(MatrixMultiplyCalculatorTest, Multiply) {
  CalculatorRunner runner("MatrixMultiplyCalculator", "", 1, 1, 1);
  Matrix* matrix = new Matrix();
  MatrixFromTextProto(kMatrixText, matrix);
  runner.MutableSidePackets()->Index(0) = Adopt(matrix);

  Matrix samples;
  MatrixFromTextProto(kSamplesText, &samples);
  Matrix expected;
  MatrixFromTextProto(kExpectedText, &expected);
  CHECK_EQ(samples.cols(), expected.cols());

  for (int i = 0; i < samples.cols(); ++i) {
    // Take a column from samples and produce a packet with just that
    // column in it as an input sample for the calculator.
    Eigen::MatrixXf* sample = new Eigen::MatrixXf(samples.block(0, i, 4, 1));
    runner.MutableInputs()->Index(0).packets.push_back(
        Adopt(sample).At(Timestamp(i)));
  }

  MP_ASSERT_OK(runner.Run());
  EXPECT_EQ(runner.MutableInputs()->Index(0).packets.size(),
            runner.Outputs().Index(0).packets.size());

  int i = 0;
  for (const Packet& output : runner.Outputs().Index(0).packets) {
    EXPECT_EQ(Timestamp(i), output.Timestamp());
    const Eigen::MatrixXf& result = output.Get<Matrix>();
    ASSERT_EQ(3, result.rows());
    EXPECT_NEAR((expected.block(0, i, 3, 1) - result).cwiseAbs().sum(), 0.0,
                1e-5);
    ++i;
  }
  EXPECT_EQ(samples.cols(), i);
}

}  // namespace
}  // namespace mediapipe
