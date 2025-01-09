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
#include <string>
#include <vector>

#include "Eigen/Core"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {

class SumTimeSeriesAcrossChannelsCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override {
    calculator_name_ = "SumTimeSeriesAcrossChannelsCalculator";
  }
};

TEST_F(SumTimeSeriesAcrossChannelsCalculatorTest, IsNoOpOnSingleChannelInputs) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 1  num_samples: 5");
  const Matrix input =
      Matrix::Random(header.num_channels(), header.num_samples());

  Test(header, {input}, header, {input});
}

TEST_F(SumTimeSeriesAcrossChannelsCalculatorTest, ConstantPacket) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 3  num_samples: 5");
  TimeSeriesHeader output_header(header);
  output_header.set_num_channels(1);

  Test(header,
       {Matrix::Constant(header.num_channels(), header.num_samples(), 1)},
       output_header,
       {Matrix::Constant(1, header.num_samples(), header.num_channels())});
}

TEST_F(SumTimeSeriesAcrossChannelsCalculatorTest, MultiplePackets) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 3  num_samples: 5");
  Matrix in(header.num_channels(), header.num_samples());
  in << 10, -1, -1, 0, 0, 20, -2, 0, 1, 0, 30, -3, 1, 0, 12;

  TimeSeriesHeader output_header(header);
  output_header.set_num_channels(1);
  Matrix out(1, header.num_samples());
  out << 60, -6, 0, 1, 12;

  Test(header, {in, 2 * in, in + Matrix::Constant(in.rows(), in.cols(), 3.5f)},
       output_header,
       {out, 2 * out,
        out + Matrix::Constant(out.rows(), out.cols(),
                               3.5 * header.num_channels())});
}

class AverageTimeSeriesAcrossChannelsCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override {
    calculator_name_ = "AverageTimeSeriesAcrossChannelsCalculator";
  }
};

TEST_F(AverageTimeSeriesAcrossChannelsCalculatorTest,
       IsNoOpOnSingleChannelInputs) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 1  num_samples: 5");
  const Matrix input =
      Matrix::Random(header.num_channels(), header.num_samples());

  Test(header, {input}, header, {input});
}

TEST_F(AverageTimeSeriesAcrossChannelsCalculatorTest, ConstantPacket) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 3  num_samples: 5");
  TimeSeriesHeader output_header(header);
  output_header.set_num_channels(1);

  Matrix input =
      Matrix::Constant(header.num_channels(), header.num_samples(), 0.0);
  input.row(0) = Matrix::Constant(1, header.num_samples(), 1.0);

  Test(
      header, {input}, output_header,
      {Matrix::Constant(1, header.num_samples(), 1.0 / header.num_channels())});
}

class SummarySaiToPitchogramCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override {
    calculator_name_ = "SummarySaiToPitchogramCalculator";
  }
};

TEST_F(SummarySaiToPitchogramCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 1  num_samples: 3");
  Matrix input(1, input_header.num_samples());
  input << 3, -9, 4;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 5.0  packet_rate: 5.0  num_channels: 3  num_samples: 1");
  Matrix output(input_header.num_samples(), 1);
  output << 3, -9, 4;

  Test(input_header, {input}, output_header, {output});
}

class ReverseChannelOrderCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "ReverseChannelOrderCalculator"; }
};

TEST_F(ReverseChannelOrderCalculatorTest, IsNoOpOnSingleChannelInputs) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 1  num_samples: 5");
  const Matrix input =
      Matrix::Random(header.num_channels(), header.num_samples());

  Test(header, {input}, header, {input});
}

TEST_F(ReverseChannelOrderCalculatorTest, SinglePacket) {
  const TimeSeriesHeader header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  num_channels: 5  num_samples: 2");
  Matrix input(header.num_channels(), header.num_samples());
  input.transpose() << 1, 2, 3, 4, 5, -1, -2, -3, -4, -5;
  Matrix output(header.num_channels(), header.num_samples());
  output.transpose() << 5, 4, 3, 2, 1, -5, -4, -3, -2, -1;

  Test(header, {input}, header, {output});
}

class FlattenPacketCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "FlattenPacketCalculator"; }
};

TEST_F(FlattenPacketCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input.transpose() << 1, 2, 3, 4, 5, -1, -2, -3, -4, -5;
  Matrix output(10, 1);
  output << 1, 2, 3, 4, 5, -1, -2, -3, -4, -5;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 10.0  packet_rate: 10.0  num_channels: 10  num_samples: 1");
  Test(input_header, {input}, output_header, {output});
}

class SubtractMeanCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "SubtractMeanCalculator"; }
};

TEST_F(SubtractMeanCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  Matrix output(input_header.num_channels(), input_header.num_samples());

  // clang-format off
  input.transpose() << 1,  0,  3, 0, 1,
                      -1, -2, -3, 4, 7;
  output.transpose() << 1,  1,  3, -2, -3,
                       -1, -1, -3,  2,  3;
  // clang-format on

  const TimeSeriesHeader output_header = input_header;
  Test(input_header, {input}, output_header, {output});
}

class SubtractMeanAcrossChannelsCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override {
    calculator_name_ = "SubtractMeanAcrossChannelsCalculator";
  }
};

TEST_F(SubtractMeanAcrossChannelsCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0 packet_rate: 10.0 num_channels: 3 num_samples: 2");
  TimeSeriesHeader output_header(input_header);
  output_header.set_num_samples(2);

  Matrix input(input_header.num_channels(), input_header.num_samples());
  Matrix output(output_header.num_channels(), output_header.num_samples());

  // clang-format off
  input.transpose() << 1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0;
  output.transpose() << 1.0 - 3.5, 2.0 - 3.5, 3.0 - 3.5,
                        4.0 - 3.5, 5.0 - 3.5, 6.0 - 3.5;
  // clang-format on

  Test(input_header, {input}, output_header, {output});
}

class DivideByMeanAcrossChannelsCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override {
    calculator_name_ = "DivideByMeanAcrossChannelsCalculator";
  }
};

TEST_F(DivideByMeanAcrossChannelsCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0 packet_rate: 10.0 num_channels: 3 num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input.transpose() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  TimeSeriesHeader output_header(input_header);
  output_header.set_num_samples(2);
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output.transpose() << 1.0 / 3.5, 2.0 / 3.5, 3.0 / 3.5, 4.0 / 3.5, 5.0 / 3.5,
      6.0 / 3.5;

  Test(input_header, {input}, output_header, {output});
}

TEST_F(DivideByMeanAcrossChannelsCalculatorTest, ReturnsOneForZeroMean) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0 packet_rate: 10.0 num_channels: 3 num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input.transpose() << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0;

  TimeSeriesHeader output_header(input_header);
  output_header.set_num_samples(2);
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output.transpose() << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  Test(input_header, {input}, output_header, {output});
}

class MeanCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "MeanCalculator"; }
};

TEST_F(MeanCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0 packet_rate: 10.0 num_channels: 3 num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input.transpose() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  TimeSeriesHeader output_header(input_header);
  output_header.set_num_samples(1);
  output_header.set_sample_rate(10.0);
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output << (1.0 + 4.0) / 2, (2.0 + 5.0) / 2, (3.0 + 6.0) / 2;

  Test(input_header, {input}, output_header, {output});
}

class StandardDeviationCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "StandardDeviationCalculator"; }
};

TEST_F(StandardDeviationCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0 packet_rate: 10.0 num_channels: 3 num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input.transpose() << 0.0, 2.0, 3.0, 4.0, 5.0, 8.0;

  TimeSeriesHeader output_header(input_header);
  output_header.set_sample_rate(10.0);
  output_header.set_num_samples(1);
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output << sqrt((pow(0.0 - 2.0, 2) + pow(4.0 - 2.0, 2)) / 2),
      sqrt((pow(2.0 - 3.5, 2) + pow(5.0 - 3.5, 2)) / 2),
      sqrt((pow(3.0 - 5.5, 2) + pow(8.0 - 5.5, 2)) / 2);

  Test(input_header, {input}, output_header, {output});
}

class CovarianceCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "CovarianceCalculator"; }
};

TEST_F(CovarianceCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0 packet_rate: 10.0 num_channels: 3 num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());

  // We'll specify in transposed form so we can write one channel at a time.
  input << 1.0, 3.0, 5.0, 9.0, -1.0, -3.0;

  TimeSeriesHeader output_header(input_header);
  output_header.set_num_samples(output_header.num_channels());
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output << 1, 2, -1, 2, 4, -2, -1, -2, 1;
  Test(input_header, {input}, output_header, {output});
}

class L2NormCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "L2NormCalculator"; }
};

TEST_F(L2NormCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 3, 5, 8, 4, 12, -15;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 1  num_samples: 3");
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output << 5, 13, 17;

  Test(input_header, {input}, output_header, {output});
}

class L2NormalizeColumnCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "L2NormalizeColumnCalculator"; }
};

TEST_F(L2NormalizeColumnCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 0.3, 0.4, 0.8, 0.5, 0.9, 0.8;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix output(output_header.num_channels(), output_header.num_samples());

  // The values in output are column-wise L2 normalized
  // e.g.
  //    |a| -> |a/sqrt(a^2 + b^2)|
  //    |b|    |b/sqrt(a^2 + b^2)|
  output << 0.51449579000473022, 0.40613847970962524, 0.70710676908493042,
      0.85749292373657227, 0.91381156444549561, 0.70710676908493042;

  Test(input_header, {input}, output_header, {output});
}

class L2NormalizeCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "L2NormalizeCalculator"; }
};

TEST_F(L2NormalizeCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 0.3, 0.4, 0.8, 0.5, 0.9, 0.8;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix output(output_header.num_channels(), output_header.num_samples());

  // The values in output are L2 normalized
  //    a -> a/sqrt(a^2 + b^2 + c^2 + ...) * sqrt(matrix.cols()*matrix.rows())
  output << 0.45661166, 0.60881555, 1.21763109, 0.76101943, 1.36983498,
      1.21763109;

  Test(input_header, {input}, output_header, {output});
}

TEST_F(L2NormalizeCalculatorTest, UnitMatrixStaysUnchanged) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 3  num_samples: 5");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
      1.0, -1.0, 1.0;

  Test(input_header, {input}, input_header, {input});
}

class PeakNormalizeCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "PeakNormalizeCalculator"; }
};

TEST_F(PeakNormalizeCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 0.3, 0.4, 0.8, 0.5, 0.9, 0.8;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output << 0.33333333, 0.44444444, 0.88888889, 0.55555556, 1.0, 0.88888889;

  Test(input_header, {input}, output_header, {output});
}

TEST_F(PeakNormalizeCalculatorTest, UnitMatrixStaysUnchanged) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 3  num_samples: 5");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
      1.0, -1.0, 1.0;

  Test(input_header, {input}, input_header, {input});
}

class ElementwiseSquareCalculatorTest
    : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "ElementwiseSquareCalculator"; }
};

TEST_F(ElementwiseSquareCalculatorTest, SinglePacket) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  input << 3, 5, 8, 4, 12, -15;

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 8000.0  packet_rate: 5.0  num_channels: 2  num_samples: 3");
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output << 9, 25, 64, 16, 144, 225;

  Test(input_header, {input}, output_header, {output});
}

class FirstHalfSlicerCalculatorTest : public BasicTimeSeriesCalculatorTestBase {
 protected:
  void SetUp() override { calculator_name_ = "FirstHalfSlicerCalculator"; }
};

TEST_F(FirstHalfSlicerCalculatorTest, SinglePacketEvenNumSamples) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  // clang-format off
  input.transpose() << 0, 1, 2, 3, 4,
                       5, 6, 7, 8, 9;
  // clang-format on

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 1");
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output.transpose() << 0, 1, 2, 3, 4;

  Test(input_header, {input}, output_header, {output});
}

TEST_F(FirstHalfSlicerCalculatorTest, SinglePacketOddNumSamples) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 3");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  // clang-format off
  input.transpose() << 0, 1, 2, 3, 4,
                       5, 6, 7, 8, 9,
                       0, 0, 0, 0, 0;
  // clang-format on

  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 1");
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output.transpose() << 0, 1, 2, 3, 4;

  Test(input_header, {input}, output_header, {output});
}

TEST_F(FirstHalfSlicerCalculatorTest, MultiplePackets) {
  const TimeSeriesHeader input_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 2");
  Matrix input(input_header.num_channels(), input_header.num_samples());
  // clang-format off
  input.transpose() << 0, 1, 2, 3, 4,
                       5, 6, 7, 8, 9;
  // clang-format on
  const TimeSeriesHeader output_header = ParseTextProtoOrDie<TimeSeriesHeader>(
      "sample_rate: 20.0  packet_rate: 10.0  num_channels: 5  num_samples: 1");
  Matrix output(output_header.num_channels(), output_header.num_samples());
  output.transpose() << 0, 1, 2, 3, 4;

  Test(input_header,
       {input, 2 * input,
        input + Matrix::Constant(input.rows(), input.cols(), 3.5f)},
       output_header,
       {output, 2 * output,
        output + Matrix::Constant(output.rows(), output.cols(), 3.5f)});
}

}  // namespace mediapipe
