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

#include <math.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/log/absl_log.h"
#include "audio/dsp/window_functions.h"
#include "mediapipe/calculators/audio/time_series_framer_calculator.pb.h"
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

const int kInitialTimestampOffsetMicroseconds = 4;
const int kGapBetweenPacketsInSeconds = 1;
const int kUniversalInputPacketSize = 50;

class TimeSeriesFramerCalculatorTest
    : public TimeSeriesCalculatorTest<TimeSeriesFramerCalculatorOptions> {
 protected:
  void SetUp() override {
    calculator_name_ = "TimeSeriesFramerCalculator";
    input_sample_rate_ = 4000.0;
    num_input_channels_ = 3;
  }

  // Returns a float value with the channel and timestamp separated by
  // an order of magnitude, for easy parsing by humans.
  float TestValue(int64_t timestamp_in_microseconds, int channel) {
    return timestamp_in_microseconds + channel / 10.0;
  }

  // Caller takes ownership of the returned value.
  Matrix* NewTestFrame(int num_channels, int num_samples,
                       double starting_timestamp_seconds) {
    auto matrix = new Matrix(num_channels, num_samples);
    for (int c = 0; c < num_channels; ++c) {
      for (int i = 0; i < num_samples; ++i) {
        int64_t timestamp = time_series_util::SecondsToSamples(
            starting_timestamp_seconds + i / input_sample_rate_,
            Timestamp::kTimestampUnitsPerSecond);
        (*matrix)(c, i) = TestValue(timestamp, c);
      }
    }
    return matrix;
  }

  // Initializes and runs the test graph.
  absl::Status Run() {
    InitializeGraph();

    FillInputHeader();
    InitializeInput();

    return RunGraph();
  }

  // Creates test input and saves a reference copy.
  void InitializeInput() {
    concatenated_input_samples_.resize(0, num_input_channels_);
    num_input_samples_ = 0;
    for (int i = 0; i < 10; ++i) {
      // This range of packet sizes was chosen such that some input
      // packets will be smaller than the output packet size and other
      // input packets will be larger.
      int packet_size = (i + 1) * 20;
      double timestamp_seconds = kInitialTimestampOffsetMicroseconds * 1.0e-6 +
                                 num_input_samples_ / input_sample_rate_;

      Matrix* data_frame =
          NewTestFrame(num_input_channels_, packet_size, timestamp_seconds);

      // Keep a reference copy of the input.
      //
      // conservativeResize() is needed here to preserve the existing
      // data.  Eigen's resize() resizes without preserving data.
      concatenated_input_samples_.conservativeResize(
          num_input_channels_, num_input_samples_ + packet_size);
      concatenated_input_samples_.rightCols(packet_size) = *data_frame;
      num_input_samples_ += packet_size;

      AppendInputPacket(data_frame, round(timestamp_seconds *
                                          Timestamp::kTimestampUnitsPerSecond));
    }

    const int frame_duration_samples = FrameDurationSamples();
    std::vector<double> window_vector;
    switch (options_.window_function()) {
      case TimeSeriesFramerCalculatorOptions::HAMMING:
        audio_dsp::HammingWindow().GetPeriodicSamples(frame_duration_samples,
                                                      &window_vector);
        break;
      case TimeSeriesFramerCalculatorOptions::HANN:
        audio_dsp::HannWindow().GetPeriodicSamples(frame_duration_samples,
                                                   &window_vector);
        break;
      case TimeSeriesFramerCalculatorOptions::NONE:
        window_vector.assign(frame_duration_samples, 1.0f);
        break;
    }

    window_ = Matrix::Ones(num_input_channels_, 1) *
              Eigen::Map<Eigen::MatrixXd>(window_vector.data(), 1,
                                          frame_duration_samples)
                  .cast<float>();
  }

  int FrameDurationSamples() {
    return time_series_util::SecondsToSamples(options_.frame_duration_seconds(),
                                              input_sample_rate_);
  }

  // Checks that the values in the framed output packets matches the
  // appropriate values from the input.
  void CheckOutputPacketValues(const Matrix& actual, int packet_num,
                               int frame_duration_samples,
                               double frame_step_samples,
                               int num_columns_to_check) {
    ASSERT_EQ(frame_duration_samples, actual.cols());
    Matrix expected = (concatenated_input_samples_
                           .block(0, round(frame_step_samples * packet_num),
                                  num_input_channels_, num_columns_to_check)
                           .array() *
                       window_.leftCols(num_columns_to_check).array())
                          .matrix();
    ExpectApproximatelyEqual(expected, actual.leftCols(num_columns_to_check));
  }

  // Checks output headers, Timestamps, and values.
  void CheckOutput() {
    const int frame_duration_samples = FrameDurationSamples();
    const double frame_step_samples =
        options_.emulate_fractional_frame_overlap()
            ? (options_.frame_duration_seconds() -
               options_.frame_overlap_seconds()) *
                  input_sample_rate_
            : frame_duration_samples -
                  time_series_util::SecondsToSamples(
                      options_.frame_overlap_seconds(), input_sample_rate_);

    TimeSeriesHeader expected_header = input().header.Get<TimeSeriesHeader>();
    expected_header.set_num_samples(frame_duration_samples);
    if (!options_.emulate_fractional_frame_overlap() ||
        frame_step_samples == round(frame_step_samples)) {
      expected_header.set_packet_rate(input_sample_rate_ / frame_step_samples);
    }
    ExpectOutputHeaderEquals(expected_header);

    int num_full_packets = output().packets.size();
    if (options_.pad_final_packet()) {
      num_full_packets -= 1;
    }

    for (int packet_num = 0; packet_num < num_full_packets; ++packet_num) {
      const Packet& packet = output().packets[packet_num];
      CheckOutputPacketValues(packet.Get<Matrix>(), packet_num,
                              frame_duration_samples, frame_step_samples,
                              frame_duration_samples);
    }

    // What is the effective time index of the final sample emitted?
    // This includes accounting for the gaps when overlap is negative.
    const int num_unique_output_samples =
        round((output().packets.size() - 1) * frame_step_samples) +
        frame_duration_samples;
    ABSL_LOG(INFO) << "packets.size()=" << output().packets.size()
                   << " frame_duration_samples=" << frame_duration_samples
                   << " frame_step_samples=" << frame_step_samples
                   << " num_input_samples_=" << num_input_samples_
                   << " num_unique_output_samples="
                   << num_unique_output_samples;
    const int num_padding_samples =
        num_unique_output_samples - num_input_samples_;
    if (options_.pad_final_packet()) {
      EXPECT_LT(num_padding_samples, frame_duration_samples);
      // If the input ended during the dropped samples between the end of
      // the last emitted frame and where the next one would begin, there
      // can be fewer unique output points than input points, even with
      // padding.
      const int max_dropped_samples =
          static_cast<int>(ceil(frame_step_samples - frame_duration_samples));
      EXPECT_GE(num_padding_samples, std::min(0, -max_dropped_samples));

      if (num_padding_samples > 0) {
        // Check the non-padded part of the final packet.
        const Matrix& final_matrix = output().packets.back().Get<Matrix>();
        CheckOutputPacketValues(final_matrix, num_full_packets,
                                frame_duration_samples, frame_step_samples,
                                frame_duration_samples - num_padding_samples);
        // Check the padded part of the final packet.
        EXPECT_EQ(
            Matrix::Zero(num_input_channels_, num_padding_samples),
            final_matrix.block(0, frame_duration_samples - num_padding_samples,
                               num_input_channels_, num_padding_samples));
      }
    } else {
      EXPECT_GT(num_padding_samples, -frame_duration_samples);
      EXPECT_LE(num_padding_samples, 0);
    }
  }

  int num_input_samples_;
  Matrix concatenated_input_samples_;
  Matrix window_;
};

TEST_F(TimeSeriesFramerCalculatorTest, IntegerSampleDurationNoOverlap) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  MP_ASSERT_OK(Run());
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest,
       IntegerSampleDurationNoOverlapHammingWindow) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_window_function(TimeSeriesFramerCalculatorOptions::HAMMING);
  MP_ASSERT_OK(Run());
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest,
       IntegerSampleDurationNoOverlapHannWindow) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_window_function(TimeSeriesFramerCalculatorOptions::HANN);
  MP_ASSERT_OK(Run());
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, IntegerSampleDurationAndOverlap) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(40.0 / input_sample_rate_);
  MP_ASSERT_OK(Run());
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, NonintegerSampleDurationAndOverlap) {
  options_.set_frame_duration_seconds(98.5 / input_sample_rate_);
  options_.set_frame_overlap_seconds(38.4 / input_sample_rate_);

  MP_ASSERT_OK(Run());
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, NegativeOverlapExactFrames) {
  // Negative overlap means to drop samples between frames.
  // 100 samples per frame plus a skip of 10 samples will be 10 full frames in
  // the 1100 input samples.
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(-10.0 / input_sample_rate_);
  MP_ASSERT_OK(Run());
  EXPECT_EQ(output().packets.size(), 10);
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, NegativeOverlapExactFramesLessSkip) {
  // 100 samples per frame plus a skip of 100 samples will be 6 full frames in
  // the 1100 input samples.
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(-100.0 / input_sample_rate_);
  MP_ASSERT_OK(Run());
  EXPECT_EQ(output().packets.size(), 6);
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, NegativeOverlapWithPadding) {
  // 150 samples per frame plus a skip of 50 samples will require some padding
  // on the sixth and last frame given 1100 sample input.
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(-100.0 / input_sample_rate_);
  MP_ASSERT_OK(Run());
  EXPECT_EQ(output().packets.size(), 6);
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, FixedFrameOverlap) {
  // Frame of 30 samples with step of 11.4 samples (rounded to 11 samples)
  // results in ceil((1100 - 30) / 11) + 1 = 99 packets.
  options_.set_frame_duration_seconds(30 / input_sample_rate_);
  options_.set_frame_overlap_seconds((30.0 - 11.4) / input_sample_rate_);
  MP_ASSERT_OK(Run());
  EXPECT_EQ(output().packets.size(), 99);
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, VariableFrameOverlap) {
  // Frame of 30 samples with step of 11.4 samples (not rounded)
  // results in ceil((1100 - 30) / 11.4) + 1 = 95 packets.
  options_.set_frame_duration_seconds(30 / input_sample_rate_);
  options_.set_frame_overlap_seconds((30 - 11.4) / input_sample_rate_);
  options_.set_emulate_fractional_frame_overlap(true);
  MP_ASSERT_OK(Run());
  EXPECT_EQ(output().packets.size(), 95);
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, VariableFrameSkip) {
  // Frame of 30 samples with step of 41.4 samples (not rounded)
  // results in ceil((1100 - 30) / 41.4) + 1 = 27 packets.
  options_.set_frame_duration_seconds(30 / input_sample_rate_);
  options_.set_frame_overlap_seconds((30 - 41.4) / input_sample_rate_);
  options_.set_emulate_fractional_frame_overlap(true);
  MP_ASSERT_OK(Run());
  EXPECT_EQ(output().packets.size(), 27);
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest, NoFinalPacketPadding) {
  options_.set_frame_duration_seconds(98.5 / input_sample_rate_);
  options_.set_pad_final_packet(false);

  MP_ASSERT_OK(Run());
  CheckOutput();
}

TEST_F(TimeSeriesFramerCalculatorTest,
       FrameRateHigherThanSampleRate_FrameDurationTooLow) {
  // Try to produce a frame rate 10 times the input sample rate by using a
  // a frame duration that is too small and covers only 0.1 samples.
  options_.set_frame_duration_seconds(1 / (10 * input_sample_rate_));
  options_.set_frame_overlap_seconds(0.0);
  EXPECT_FALSE(Run().ok());
}

TEST_F(TimeSeriesFramerCalculatorTest,
       FrameRateHigherThanSampleRate_FrameStepTooLow) {
  // Try to produce a frame rate 10 times the input sample rate by using
  // a frame overlap that is too high and produces frame steps (difference
  // between duration and overlap) of 0.1 samples.
  options_.set_frame_duration_seconds(10.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(9.9 / input_sample_rate_);
  EXPECT_FALSE(Run().ok());
}

// A simple test class to do windowing sanity checks. Tests from this
// class input a single packet of all ones, and check the average
// value of the single output packet. This is useful as a sanity check
// that the correct windows are applied.
class TimeSeriesFramerCalculatorWindowingSanityTest
    : public TimeSeriesFramerCalculatorTest {
 protected:
  void SetUp() override {
    TimeSeriesFramerCalculatorTest::SetUp();
    num_input_channels_ = 1;
  }

  void RunAndTestSinglePacketAverage(float expected_average) {
    options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
    InitializeGraph();
    FillInputHeader();
    AppendInputPacket(new Matrix(Matrix::Ones(1, FrameDurationSamples())),
                      kInitialTimestampOffsetMicroseconds);
    MP_ASSERT_OK(RunGraph());
    ASSERT_EQ(1, output().packets.size());
    ASSERT_NEAR(expected_average * FrameDurationSamples(),
                output().packets[0].Get<Matrix>().sum(), 1e-5);
  }
};

TEST_F(TimeSeriesFramerCalculatorWindowingSanityTest, NoWindowSanityCheck) {
  RunAndTestSinglePacketAverage(1.0f);
}

TEST_F(TimeSeriesFramerCalculatorWindowingSanityTest,
       HammingWindowSanityCheck) {
  options_.set_window_function(TimeSeriesFramerCalculatorOptions::HAMMING);
  RunAndTestSinglePacketAverage(0.54f);
}

TEST_F(TimeSeriesFramerCalculatorWindowingSanityTest, HannWindowSanityCheck) {
  options_.set_window_function(TimeSeriesFramerCalculatorOptions::HANN);
  RunAndTestSinglePacketAverage(0.5f);
}

// A simple test class that checks the local packet time stamp. This class
// generate a series of packets with and without gaps between packets and tests
// the behavior with cumulative timestamping and local packet timestamping.
class TimeSeriesFramerCalculatorTimestampingTest
    : public TimeSeriesFramerCalculatorTest {
 protected:
  // Creates test input and saves a reference copy.
  void InitializeInputForTimeStampingTest() {
    concatenated_input_samples_.resize(0, num_input_channels_);
    num_input_samples_ = 0;
    for (int i = 0; i < 10; ++i) {
      // This range of packet sizes was chosen such that some input
      // packets will be smaller than the output packet size and other
      // input packets will be larger.
      int packet_size = kUniversalInputPacketSize;
      double timestamp_seconds = kInitialTimestampOffsetMicroseconds * 1.0e-6 +
                                 num_input_samples_ / input_sample_rate_;
      if (options_.use_local_timestamp()) {
        timestamp_seconds += kGapBetweenPacketsInSeconds * i;
      }

      Matrix* data_frame =
          NewTestFrame(num_input_channels_, packet_size, timestamp_seconds);

      AppendInputPacket(data_frame, round(timestamp_seconds *
                                          Timestamp::kTimestampUnitsPerSecond));
      num_input_samples_ += packet_size;
    }
  }

  void CheckOutputTimestamps() {
    int num_full_packets = output().packets.size();
    if (options_.pad_final_packet()) {
      num_full_packets -= 1;
    }

    int64_t num_samples = 0;
    for (int packet_num = 0; packet_num < num_full_packets; ++packet_num) {
      const Packet& packet = output().packets[packet_num];
      num_samples += FrameDurationSamples();
      double expected_timestamp =
          options_.use_local_timestamp()
              ? GetExpectedLocalTimestampForSample(num_samples - 1)
              : GetExpectedCumulativeTimestamp(num_samples - 1);
      ASSERT_NEAR(packet.Timestamp().Seconds(), expected_timestamp, 1e-10);
    }
  }

  absl::Status RunTimestampTest() {
    InitializeGraph();
    InitializeInputForTimeStampingTest();
    FillInputHeader();
    return RunGraph();
  }

 private:
  // Returns the timestamp in seconds based on local timestamping.
  double GetExpectedLocalTimestampForSample(int sample_index) {
    return kInitialTimestampOffsetMicroseconds * 1.0e-6 +
           sample_index / input_sample_rate_ +
           (sample_index / kUniversalInputPacketSize) *
               kGapBetweenPacketsInSeconds;
  }

  // Returns the timestamp inseconds based on cumulative timestamping.
  double GetExpectedCumulativeTimestamp(int sample_index) {
    return kInitialTimestampOffsetMicroseconds * 1.0e-6 +
           sample_index / FrameDurationSamples() * FrameDurationSamples() /
               input_sample_rate_;
  }
};

TEST_F(TimeSeriesFramerCalculatorTimestampingTest, UseLocalTimeStamp) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_use_local_timestamp(true);

  MP_ASSERT_OK(RunTimestampTest());
  CheckOutputTimestamps();
}

TEST_F(TimeSeriesFramerCalculatorTimestampingTest, UseCumulativeTimeStamp) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_use_local_timestamp(false);

  MP_ASSERT_OK(RunTimestampTest());
  CheckOutputTimestamps();
}

}  // namespace
}  // namespace mediapipe
