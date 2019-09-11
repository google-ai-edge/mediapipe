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

#include "mediapipe/calculators/audio/rational_factor_resample_calculator.h"

#include <math.h>

#include <algorithm>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "audio/dsp/signal_vector_util.h"
#include "mediapipe/calculators/audio/rational_factor_resample_calculator.pb.h"
#include "mediapipe/framework//tool/validate_type.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {
namespace {

const int kInitialTimestampOffsetMilliseconds = 4;

class RationalFactorResampleCalculatorTest
    : public TimeSeriesCalculatorTest<RationalFactorResampleCalculatorOptions> {
 protected:
  void SetUp() override {
    calculator_name_ = "RationalFactorResampleCalculator";
    input_sample_rate_ = 4000.0;
    num_input_channels_ = 3;
  }

  // Expects two vectors whose lengths are almost the same and whose
  // elements are equal (for indices that are present in both).
  //
  // This is useful because the resampler doesn't make precise
  // guarantees about its output size.
  void ExpectVectorMostlyFloatEq(const std::vector<float>& expected,
                                 const std::vector<float>& actual) {
    // Lengths should be close, but don't have to be equal.
    ASSERT_NEAR(expected.size(), actual.size(), 1);
    for (int i = 0; i < std::min(expected.size(), actual.size()); ++i) {
      EXPECT_FLOAT_EQ(expected[i], actual[i]) << " where i=" << i << ".";
    }
  }

  // Returns a float value with the sample, channel, and timestamp
  // separated by a few orders of magnitude, for easy parsing by
  // humans.
  double TestValue(int sample, int channel, int timestamp_in_microseconds) {
    return timestamp_in_microseconds * 100.0 + sample + channel / 10.0;
  }

  // Caller takes ownership of the returned value.
  Matrix* NewTestFrame(int num_channels, int num_samples, int timestamp) {
    auto matrix = new Matrix(num_channels, num_samples);
    for (int c = 0; c < num_channels; ++c) {
      for (int i = 0; i < num_samples; ++i) {
        (*matrix)(c, i) = TestValue(i, c, timestamp);
      }
    }
    return matrix;
  }

  // Initializes and runs the test graph.
  ::mediapipe::Status Run(double output_sample_rate) {
    options_.set_target_sample_rate(output_sample_rate);
    InitializeGraph();

    FillInputHeader();
    concatenated_input_samples_.resize(num_input_channels_, 0);
    num_input_samples_ = 0;
    for (int i = 0; i < 5; ++i) {
      int packet_size = (i + 1) * 10;
      int timestamp = kInitialTimestampOffsetMilliseconds +
                      num_input_samples_ / input_sample_rate_ *
                          Timestamp::kTimestampUnitsPerSecond;
      Matrix* data_frame =
          NewTestFrame(num_input_channels_, packet_size, timestamp);

      // Keep a reference copy of the input.
      //
      // conservativeResize() is needed here to preserve the existing
      // data.  Eigen's resize() resizes without preserving data.
      concatenated_input_samples_.conservativeResize(
          num_input_channels_, num_input_samples_ + packet_size);
      concatenated_input_samples_.rightCols(packet_size) = *data_frame;
      num_input_samples_ += packet_size;

      AppendInputPacket(data_frame, timestamp);
    }

    return RunGraph();
  }

  void CheckOutputLength(double output_sample_rate) {
    double factor = output_sample_rate / input_sample_rate_;

    int num_output_samples = 0;
    for (const Packet& packet : output().packets) {
      num_output_samples += packet.Get<Matrix>().cols();
    }

    // The exact number of expected samples may vary based on the implementation
    // of the resampler since the exact value is not an integer.
    // TODO: Reduce this offset to + 1 once cl/185829520 is submitted.
    const double expected_num_output_samples = num_input_samples_ * factor;
    EXPECT_LE(ceil(expected_num_output_samples), num_output_samples);
    EXPECT_GE(ceil(expected_num_output_samples) + 11, num_output_samples);
  }

  // Checks that output timestamps are consistent with the
  // output_sample_rate and output packet sizes.
  void CheckOutputPacketTimestamps(double output_sample_rate) {
    int num_output_samples = 0;
    for (const Packet& packet : output().packets) {
      const int expected_timestamp = kInitialTimestampOffsetMilliseconds +
                                     num_output_samples / output_sample_rate *
                                         Timestamp::kTimestampUnitsPerSecond;
      EXPECT_NEAR(expected_timestamp, packet.Timestamp().Value(), 1);
      num_output_samples += packet.Get<Matrix>().cols();
    }
  }

  // Checks that output values from the calculator (which resamples
  // packet-by-packet) are consistent with resampling the entire
  // signal at once.
  void CheckOutputValues(double output_sample_rate) {
    for (int i = 0; i < num_input_channels_; ++i) {
      auto verification_resampler =
          RationalFactorResampleCalculator::TestAccess::ResamplerFromOptions(
              input_sample_rate_, output_sample_rate, options_);

      std::vector<float> input_data;
      for (int j = 0; j < num_input_samples_; ++j) {
        input_data.push_back(concatenated_input_samples_(i, j));
      }
      std::vector<float> expected_resampled_data;
      std::vector<float> temp;
      verification_resampler->ProcessSamples(input_data, &temp);
      audio_dsp::VectorAppend(&expected_resampled_data, temp);
      verification_resampler->Flush(&temp);
      audio_dsp::VectorAppend(&expected_resampled_data, temp);
      std::vector<float> actual_resampled_data;
      for (const Packet& packet : output().packets) {
        Matrix output_frame_row = packet.Get<Matrix>().row(i);
        actual_resampled_data.insert(
            actual_resampled_data.end(), &output_frame_row(0),
            &output_frame_row(0) + output_frame_row.cols());
      }

      ExpectVectorMostlyFloatEq(expected_resampled_data, actual_resampled_data);
    }
  }

  void CheckOutputHeaders(double output_sample_rate) {
    const TimeSeriesHeader& output_header =
        output().header.Get<TimeSeriesHeader>();
    TimeSeriesHeader expected_header;
    expected_header.set_sample_rate(output_sample_rate);
    expected_header.set_num_channels(num_input_channels_);
    EXPECT_THAT(output_header, mediapipe::EqualsProto(expected_header));
  }

  void CheckOutput(double output_sample_rate) {
    CheckOutputLength(output_sample_rate);
    CheckOutputPacketTimestamps(output_sample_rate);
    CheckOutputValues(output_sample_rate);
    CheckOutputHeaders(output_sample_rate);
  }

  void CheckOutputUnchanged() {
    for (int i = 0; i < num_input_channels_; ++i) {
      std::vector<float> expected_resampled_data;
      for (int j = 0; j < num_input_samples_; ++j) {
        expected_resampled_data.push_back(concatenated_input_samples_(i, j));
      }
      std::vector<float> actual_resampled_data;
      for (const Packet& packet : output().packets) {
        Matrix output_frame_row = packet.Get<Matrix>().row(i);
        actual_resampled_data.insert(
            actual_resampled_data.end(), &output_frame_row(0),
            &output_frame_row(0) + output_frame_row.cols());
      }
      ExpectVectorMostlyFloatEq(expected_resampled_data, actual_resampled_data);
    }
  }

  int num_input_samples_;
  Matrix concatenated_input_samples_;
};

TEST_F(RationalFactorResampleCalculatorTest, Upsample) {
  const double kUpsampleRate = input_sample_rate_ * 1.9;
  MP_ASSERT_OK(Run(kUpsampleRate));
  CheckOutput(kUpsampleRate);
}

TEST_F(RationalFactorResampleCalculatorTest, Downsample) {
  const double kDownsampleRate = input_sample_rate_ / 1.9;
  MP_ASSERT_OK(Run(kDownsampleRate));
  CheckOutput(kDownsampleRate);
}

TEST_F(RationalFactorResampleCalculatorTest, UsesRationalFactorResampler) {
  const double kUpsampleRate = input_sample_rate_ * 2;
  MP_ASSERT_OK(Run(kUpsampleRate));
  CheckOutput(kUpsampleRate);
}

TEST_F(RationalFactorResampleCalculatorTest, PassthroughIfSampleRateUnchanged) {
  const double kUpsampleRate = input_sample_rate_;
  MP_ASSERT_OK(Run(kUpsampleRate));
  CheckOutputUnchanged();
}

TEST_F(RationalFactorResampleCalculatorTest, FailsOnBadTargetRate) {
  ASSERT_FALSE(Run(-999.9).ok());  // Invalid output sample rate.
}

TEST_F(RationalFactorResampleCalculatorTest, DoesNotDieOnEmptyInput) {
  options_.set_target_sample_rate(input_sample_rate_);
  InitializeGraph();
  FillInputHeader();
  MP_ASSERT_OK(RunGraph());
  EXPECT_TRUE(output().packets.empty());
}

}  // anonymous namespace
}  // namespace mediapipe
