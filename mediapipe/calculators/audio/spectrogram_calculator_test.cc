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

#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "audio/dsp/number_util.h"
#include "mediapipe/calculators/audio/spectrogram_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {
namespace {

const int kInitialTimestampOffsetMicroseconds = 4;

class SpectrogramCalculatorTest
    : public TimeSeriesCalculatorTest<SpectrogramCalculatorOptions> {
 protected:
  void SetUp() override {
    calculator_name_ = "SpectrogramCalculator";
    input_sample_rate_ = 4000.0;
    num_input_channels_ = 1;
  }

  // Initializes and runs the test graph.
  ::mediapipe::Status Run() {
    // Now that options are set, we can set up some internal constants.
    frame_duration_samples_ =
        round(options_.frame_duration_seconds() * input_sample_rate_);
    frame_step_samples_ =
        frame_duration_samples_ -
        round(options_.frame_overlap_seconds() * input_sample_rate_);
    // The magnitude of the 0th FFT bin (DC) should be sum(input.*window);
    // for an input identically 1.0, this is just sum(window).  The average
    // value of our Hann window is 0.5, hence this is the expected squared-
    // magnitude output value in the DC bin for constant input of 1.0.
    expected_dc_squared_magnitude_ =
        pow((static_cast<float>(frame_duration_samples_) * 0.5), 2.0);

    return RunGraph();
  }

  // Creates test multichannel input with specified packet sizes and containing
  // a constant-frequency sinusoid that maintains phase between adjacent
  // packets.
  void SetupCosineInputPackets(const std::vector<int>& packet_sizes_samples,
                               float cosine_frequency_hz) {
    int total_num_input_samples = 0;
    for (int packet_size_samples : packet_sizes_samples) {
      double packet_start_time_seconds =
          kInitialTimestampOffsetMicroseconds * 1e-6 +
          total_num_input_samples / input_sample_rate_;
      double packet_end_time_seconds =
          packet_start_time_seconds + packet_size_samples / input_sample_rate_;
      double angular_freq = 2 * M_PI * cosine_frequency_hz;
      Matrix* packet_data =
          new Matrix(num_input_channels_, packet_size_samples);
      // Use Eigen's vectorized cos() function to fill the vector with a
      // sinusoid of appropriate frequency & phase.
      for (int i = 0; i < num_input_channels_; i++) {
        packet_data->row(i) =
            Eigen::ArrayXf::LinSpaced(packet_size_samples,
                                      packet_start_time_seconds * angular_freq,
                                      packet_end_time_seconds * angular_freq)
                .cos()
                .transpose();
      }
      int64 input_timestamp = round(packet_start_time_seconds *
                                    Timestamp::kTimestampUnitsPerSecond);
      AppendInputPacket(packet_data, input_timestamp);
      total_num_input_samples += packet_size_samples;
    }
  }

  // Setup a sequence of input packets of specified sizes, each filled
  // with samples of 1.0.
  void SetupConstantInputPackets(const std::vector<int>& packet_sizes_samples) {
    // A 0 Hz cosine is identically 1.0 for all samples.
    SetupCosineInputPackets(packet_sizes_samples, 0.0);
  }

  // Setup a sequence of input packets of specified sizes, each containing a
  // single sample of 1.0 at a specified offset.
  void SetupImpulseInputPackets(
      const std::vector<int>& packet_sizes_samples,
      const std::vector<int>& impulse_offsets_samples) {
    int total_num_input_samples = 0;
    for (int i = 0; i < packet_sizes_samples.size(); ++i) {
      double packet_start_time_seconds =
          kInitialTimestampOffsetMicroseconds * 1e-6 +
          total_num_input_samples / input_sample_rate_;
      int64 input_timestamp = round(packet_start_time_seconds *
                                    Timestamp::kTimestampUnitsPerSecond);
      std::unique_ptr<Matrix> impulse(
          new Matrix(Matrix::Zero(1, packet_sizes_samples[i])));
      (*impulse)(0, impulse_offsets_samples[i]) = 1.0;
      AppendInputPacket(impulse.release(), input_timestamp);
      total_num_input_samples += packet_sizes_samples[i];
    }
  }

  // Creates test multichannel input with specified packet sizes and containing
  // constant input packets for the even channels and constant-frequency
  // sinusoid that maintains phase between adjacent packets for the odd
  // channels.
  void SetupMultichannelInputPackets(
      const std::vector<int>& packet_sizes_samples, float cosine_frequency_hz) {
    int total_num_input_samples = 0;
    for (int packet_size_samples : packet_sizes_samples) {
      double packet_start_time_seconds =
          kInitialTimestampOffsetMicroseconds * 1e-6 +
          total_num_input_samples / input_sample_rate_;
      double packet_end_time_seconds =
          packet_start_time_seconds + packet_size_samples / input_sample_rate_;
      double angular_freq;
      Matrix* packet_data =
          new Matrix(num_input_channels_, packet_size_samples);
      // Use Eigen's vectorized cos() function to fill the vector with a
      // sinusoid of appropriate frequency & phase.
      for (int i = 0; i < num_input_channels_; i++) {
        if (i % 2 == 0) {
          angular_freq = 0;
        } else {
          angular_freq = 2 * M_PI * cosine_frequency_hz;
        }
        packet_data->row(i) =
            Eigen::ArrayXf::LinSpaced(packet_size_samples,
                                      packet_start_time_seconds * angular_freq,
                                      packet_end_time_seconds * angular_freq)
                .cos()
                .transpose();
      }
      int64 input_timestamp = round(packet_start_time_seconds *
                                    Timestamp::kTimestampUnitsPerSecond);
      AppendInputPacket(packet_data, input_timestamp);
      total_num_input_samples += packet_size_samples;
    }
  }

  // Return vector of the numbers of frames in each output packet.
  std::vector<int> OutputFramesPerPacket() {
    std::vector<int> frame_counts;
    for (const Packet& packet : output().packets) {
      const Matrix& matrix = packet.Get<Matrix>();
      frame_counts.push_back(matrix.cols());
    }
    return frame_counts;
  }

  // Checks output headers and Timestamps.
  void CheckOutputHeadersAndTimestamps() {
    const int fft_size = audio_dsp::NextPowerOfTwo(frame_duration_samples_);

    TimeSeriesHeader expected_header = input().header.Get<TimeSeriesHeader>();
    expected_header.set_num_channels(fft_size / 2 + 1);
    // The output header sample rate should depend on the output frame step.
    expected_header.set_sample_rate(input_sample_rate_ / frame_step_samples_);
    // SpectrogramCalculator stores the sample rate of the input in
    // the TimeSeriesHeader.
    expected_header.set_audio_sample_rate(input_sample_rate_);
    // We expect the output header to have num_samples and packet_rate unset.
    expected_header.clear_num_samples();
    expected_header.clear_packet_rate();
    if (!options_.allow_multichannel_input()) {
      ExpectOutputHeaderEquals(expected_header);
    } else {
      EXPECT_THAT(output()
                      .header.template Get<MultiStreamTimeSeriesHeader>()
                      .time_series_header(),
                  mediapipe::EqualsProto(expected_header));
      EXPECT_THAT(output()
                      .header.template Get<MultiStreamTimeSeriesHeader>()
                      .num_streams(),
                  num_input_channels_);
    }

    int cumulative_output_frames = 0;
    // The timestamps coming out of the spectrogram correspond to the
    // middle of the first frame's window, hence frame_duration_samples_/2
    // term.  We use frame_duration_samples_ because that is how it is
    // actually quantized inside spectrogram.
    const double packet_timestamp_offset_seconds =
        kInitialTimestampOffsetMicroseconds * 1e-6;
    const double frame_step_seconds = frame_step_samples_ / input_sample_rate_;

    Timestamp initial_timestamp = Timestamp::Unstarted();

    for (const Packet& packet : output().packets) {
      // This is the timestamp we expect based on how the spectrogram should
      // behave (advancing by one step's worth of input samples each frame).
      const double expected_timestamp_seconds =
          packet_timestamp_offset_seconds +
          cumulative_output_frames * frame_step_seconds;
      const int64 expected_timestamp_ticks =
          expected_timestamp_seconds * Timestamp::kTimestampUnitsPerSecond;
      EXPECT_EQ(expected_timestamp_ticks, packet.Timestamp().Value());
      // Accept the timestamp of the first packet as the baseline for checking
      // the remainder.
      if (initial_timestamp == Timestamp::Unstarted()) {
        initial_timestamp = packet.Timestamp();
      }
      // Also check that the timestamp is consistent with the sample_rate
      // in the output stream's TimeSeriesHeader.
      EXPECT_TRUE(time_series_util::LogWarningIfTimestampIsInconsistent(
          packet.Timestamp(), initial_timestamp, cumulative_output_frames,
          expected_header.sample_rate()));
      if (!options_.allow_multichannel_input()) {
        if (options_.output_type() == SpectrogramCalculatorOptions::COMPLEX) {
          const Eigen::MatrixXcf& matrix = packet.Get<Eigen::MatrixXcf>();
          cumulative_output_frames += matrix.cols();
        } else {
          const Matrix& matrix = packet.Get<Matrix>();
          cumulative_output_frames += matrix.cols();
        }
      } else {
        if (options_.output_type() == SpectrogramCalculatorOptions::COMPLEX) {
          const Eigen::MatrixXcf& matrix =
              packet.Get<std::vector<Eigen::MatrixXcf>>().at(0);
          cumulative_output_frames += matrix.cols();
        } else {
          const Matrix& matrix = packet.Get<std::vector<Matrix>>().at(0);
          cumulative_output_frames += matrix.cols();
        }
      }
    }
  }

  // Verify that the bin corresponding to the specified frequency
  // is the largest one in one particular frame of a single packet.
  void CheckPeakFrequencyInPacketFrame(const Packet& packet, int frame,
                                       float frequency) {
    const int fft_size = audio_dsp::NextPowerOfTwo(frame_duration_samples_);
    const int target_bin =
        round((frequency / input_sample_rate_) * static_cast<float>(fft_size));

    const Matrix& matrix = packet.Get<Matrix>();
    // Stop here if the requested frame is not in this packet.
    ASSERT_GT(matrix.cols(), frame);

    int actual_largest_bin;
    matrix.col(frame).maxCoeff(&actual_largest_bin);
    EXPECT_EQ(actual_largest_bin, target_bin);
  }

  // Verify that the bin corresponding to the specified frequency
  // is the largest one in one particular frame of a single spectrogram Matrix.
  void CheckPeakFrequencyInMatrix(const Matrix& matrix, int frame,
                                  float frequency) {
    const int fft_size = audio_dsp::NextPowerOfTwo(frame_duration_samples_);
    const int target_bin =
        round((frequency / input_sample_rate_) * static_cast<float>(fft_size));

    // Stop here if the requested frame is not in this packet.
    ASSERT_GT(matrix.cols(), frame);

    int actual_largest_bin;
    matrix.col(frame).maxCoeff(&actual_largest_bin);
    EXPECT_EQ(actual_largest_bin, target_bin);
  }

  int frame_duration_samples_;
  int frame_step_samples_;
  // Expected DC output for a window of pure 1.0, set when window length
  // is set.
  float expected_dc_squared_magnitude_;
};

TEST_F(SpectrogramCalculatorTest, IntegerFrameDurationNoOverlap) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(0.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  const std::vector<int> input_packet_sizes = {500, 200};
  const std::vector<int> expected_output_packet_sizes = {5, 2};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, IntegerFrameDurationSomeOverlap) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  const std::vector<int> input_packet_sizes = {500, 200};
  // complete_output_frames = 1 + floor((input_samples - window_length)/step)
  //   = 1 + floor((500 - 100)/40) = 1 + 10 = 11 for the first packet
  //   = 1 + floor((700 - 100)/40) = 1 + 15 = 16 for the whole stream
  // so expect 16 - 11 = 5 in the second packet.
  const std::vector<int> expected_output_packet_sizes = {11, 5};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, NonintegerFrameDurationAndOverlap) {
  options_.set_frame_duration_seconds(98.5 / input_sample_rate_);
  options_.set_frame_overlap_seconds(58.4 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  const std::vector<int> input_packet_sizes = {500, 200};
  // now frame_duration_samples will be 99 (rounded), and frame_step_samples
  // will be (99-58) = 41, so the first packet of 500 samples will generate
  // 1 + floor(500-99)/41 = 10 samples.
  const std::vector<int> expected_output_packet_sizes = {10, 5};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, ShortInitialPacketNoOverlap) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(0.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  const std::vector<int> input_packet_sizes = {90, 100, 110};
  // The first input packet is too small to generate any frames,
  // but zero-length packets would result in a timestamp monotonicity
  // violation, so they are suppressed.  Thus, only the second and third
  // input packets generate output packets.
  const std::vector<int> expected_output_packet_sizes = {1, 2};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, TrailingSamplesNoPad) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  const std::vector<int> input_packet_sizes = {140, 90};
  const std::vector<int> expected_output_packet_sizes = {2, 2};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, NoTrailingSamplesWithPad) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(true);
  const std::vector<int> input_packet_sizes = {140, 80};
  const std::vector<int> expected_output_packet_sizes = {2, 2};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, TrailingSamplesWithPad) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(true);
  const std::vector<int> input_packet_sizes = {140, 90};
  // In contrast to NoTrailingSamplesWithPad and TrailingSamplesNoPad,
  // this time we get an extra frame in an extra final packet.
  const std::vector<int> expected_output_packet_sizes = {2, 2, 1};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, VeryShortInputWillPad) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(true);
  const std::vector<int> input_packet_sizes = {30};
  const std::vector<int> expected_output_packet_sizes = {1};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, VeryShortInputZeroOutputFramesIfNoPad) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  const std::vector<int> input_packet_sizes = {90};
  const std::vector<int> expected_output_packet_sizes = {};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, DCSignalIsPeakBin) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  const std::vector<int> input_packet_sizes = {140};  // Gives 2 output frames.

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  const float dc_frequency_hz = 0.0;
  CheckPeakFrequencyInPacketFrame(output().packets[0], 0, dc_frequency_hz);
  CheckPeakFrequencyInPacketFrame(output().packets[0], 1, dc_frequency_hz);
}

TEST_F(SpectrogramCalculatorTest, A440ToneIsPeakBin) {
  const std::vector<int> input_packet_sizes = {
      460};  // 100 + 9*40 for 10 frames.
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  InitializeGraph();
  FillInputHeader();
  const float tone_frequency_hz = 440.0;
  SetupCosineInputPackets(input_packet_sizes, tone_frequency_hz);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  int num_output_frames = output().packets[0].Get<Matrix>().cols();
  for (int frame = 0; frame < num_output_frames; ++frame) {
    CheckPeakFrequencyInPacketFrame(output().packets[0], frame,
                                    tone_frequency_hz);
  }
}

TEST_F(SpectrogramCalculatorTest, SquaredMagnitudeOutputLooksRight) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_output_type(SpectrogramCalculatorOptions::SQUARED_MAGNITUDE);
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_FLOAT_EQ(output().packets[0].Get<Matrix>()(0, 0),
                  expected_dc_squared_magnitude_);
}

TEST_F(SpectrogramCalculatorTest, DefaultOutputIsSquaredMagnitude) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  // Let the output_type be its default
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_FLOAT_EQ(output().packets[0].Get<Matrix>()(0, 0),
                  expected_dc_squared_magnitude_);
}

TEST_F(SpectrogramCalculatorTest, LinearMagnitudeOutputLooksRight) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_output_type(SpectrogramCalculatorOptions::LINEAR_MAGNITUDE);
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_FLOAT_EQ(output().packets[0].Get<Matrix>()(0, 0),
                  std::sqrt(expected_dc_squared_magnitude_));
}

TEST_F(SpectrogramCalculatorTest, DbMagnitudeOutputLooksRight) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_output_type(SpectrogramCalculatorOptions::DECIBELS);
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_FLOAT_EQ(output().packets[0].Get<Matrix>()(0, 0),
                  10.0 * std::log10(expected_dc_squared_magnitude_));
}

TEST_F(SpectrogramCalculatorTest, OutputScalingLooksRight) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_output_type(SpectrogramCalculatorOptions::DECIBELS);
  double output_scale = 2.5;
  options_.set_output_scale(output_scale);
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_FLOAT_EQ(
      output().packets[0].Get<Matrix>()(0, 0),
      output_scale * 10.0 * std::log10(expected_dc_squared_magnitude_));
}

TEST_F(SpectrogramCalculatorTest, ComplexOutputLooksRight) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_output_type(SpectrogramCalculatorOptions::COMPLEX);
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Setup packets with DC input (non-zero constant value).
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_FLOAT_EQ(std::norm(output().packets[0].Get<Eigen::MatrixXcf>()(0, 0)),
                  expected_dc_squared_magnitude_);
}

TEST_F(SpectrogramCalculatorTest, ComplexOutputLooksRightForImpulses) {
  const int frame_size_samples = 100;
  options_.set_frame_duration_seconds(frame_size_samples / input_sample_rate_);
  options_.set_frame_overlap_seconds(0.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  options_.set_output_type(SpectrogramCalculatorOptions::COMPLEX);
  const std::vector<int> input_packet_sizes = {frame_size_samples,
                                               frame_size_samples};
  const std::vector<int> input_packet_impulse_offsets = {49, 50};

  InitializeGraph();
  FillInputHeader();

  // Make two impulse packets offset one sample from each other
  SetupImpulseInputPackets(input_packet_sizes, input_packet_impulse_offsets);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  const int num_buckets =
      (audio_dsp::NextPowerOfTwo(frame_size_samples) / 2) + 1;
  const float precision = 0.01f;
  auto norm_fn = [](const std::complex<float>& cf) { return std::norm(cf); };

  // Both impulses should have (approximately) constant power across all
  // frequency bins
  EXPECT_TRUE(output()
                  .packets[0]
                  .Get<Eigen::MatrixXcf>()
                  .unaryExpr(norm_fn)
                  .isApproxToConstant(1.0f, precision));
  EXPECT_TRUE(output()
                  .packets[1]
                  .Get<Eigen::MatrixXcf>()
                  .unaryExpr(norm_fn)
                  .isApproxToConstant(1.0f, precision));

  // Because the second Packet's impulse is delayed by exactly one sample with
  // respect to the first Packet's impulse, the second impulse should have
  // greater phase, and in the highest frequency bin, the real part should
  // (approximately) flip sign from the first Packet to the second
  EXPECT_LT(std::arg(output().packets[0].Get<Eigen::MatrixXcf>()(1, 0)),
            std::arg(output().packets[1].Get<Eigen::MatrixXcf>()(1, 0)));
  const float highest_bucket_real_ratio =
      output().packets[0].Get<Eigen::MatrixXcf>()(num_buckets - 1, 0).real() /
      output().packets[1].Get<Eigen::MatrixXcf>()(num_buckets - 1, 0).real();
  EXPECT_NEAR(highest_bucket_real_ratio, -1.0f, precision);
}

TEST_F(SpectrogramCalculatorTest, SquaredMagnitudeOutputLooksRightForNonDC) {
  const int frame_size_samples = 100;
  options_.set_frame_duration_seconds(frame_size_samples / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_output_type(SpectrogramCalculatorOptions::SQUARED_MAGNITUDE);
  const std::vector<int> input_packet_sizes = {140};

  InitializeGraph();
  FillInputHeader();
  // Make the tone have an integral number of cycles within the window
  const int target_bin = 16;
  const int fft_size = audio_dsp::NextPowerOfTwo(frame_size_samples);
  const float tone_frequency_hz = target_bin * (input_sample_rate_ / fft_size);
  SetupCosineInputPackets(input_packet_sizes, tone_frequency_hz);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  // For a non-DC bin, the magnitude will be split between positive and
  // negative frequency bins, so it should about be half-magnitude
  // = quarter-power.
  // It's not quite exact because of the interference from the hann(100)
  // spread from the negative-frequency half.
  EXPECT_GT(output().packets[0].Get<Matrix>()(target_bin, 0),
            0.98 * expected_dc_squared_magnitude_ / 4.0);
  EXPECT_LT(output().packets[0].Get<Matrix>()(target_bin, 0),
            1.02 * expected_dc_squared_magnitude_ / 4.0);
}

TEST_F(SpectrogramCalculatorTest, ZeroOutputsForZeroInputsWithPaddingEnabled) {
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(true);
  const std::vector<int> input_packet_sizes = {};
  const std::vector<int> expected_output_packet_sizes = {};

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(OutputFramesPerPacket(), expected_output_packet_sizes);
}

TEST_F(SpectrogramCalculatorTest, NumChannelsIsRight) {
  const std::vector<int> input_packet_sizes = {460};
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  options_.set_allow_multichannel_input(true);
  num_input_channels_ = 3;
  InitializeGraph();
  FillInputHeader();
  const float tone_frequency_hz = 440.0;
  SetupCosineInputPackets(input_packet_sizes, tone_frequency_hz);
  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  EXPECT_EQ(output().packets[0].Get<std::vector<Matrix>>().size(),
            num_input_channels_);
}

TEST_F(SpectrogramCalculatorTest, NumSamplesAndPacketRateAreCleared) {
  num_input_samples_ = 500;
  input_packet_rate_ = 1.0;
  const std::vector<int> input_packet_sizes = {num_input_samples_};
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(0.0);
  options_.set_pad_final_packet(false);

  InitializeGraph();
  FillInputHeader();
  SetupConstantInputPackets(input_packet_sizes);

  MP_ASSERT_OK(Run());

  const TimeSeriesHeader& output_header =
      output().header.Get<TimeSeriesHeader>();
  EXPECT_FALSE(output_header.has_num_samples());
  EXPECT_FALSE(output_header.has_packet_rate());
}

TEST_F(SpectrogramCalculatorTest, MultichannelSpectrogramSizesAreRight) {
  const std::vector<int> input_packet_sizes = {420};  // less than 10 frames
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  options_.set_allow_multichannel_input(true);
  num_input_channels_ = 10;
  InitializeGraph();
  FillInputHeader();
  const float tone_frequency_hz = 440.0;
  SetupCosineInputPackets(input_packet_sizes, tone_frequency_hz);
  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  auto spectrograms = output().packets[0].Get<std::vector<Matrix>>();
  EXPECT_FLOAT_EQ(spectrograms.size(), num_input_channels_);
  int spectrogram_num_rows = spectrograms[0].rows();
  int spectrogram_num_cols = spectrograms[0].cols();
  for (int i = 1; i < num_input_channels_; i++) {
    EXPECT_EQ(spectrogram_num_rows, spectrograms[i].rows());
    EXPECT_EQ(spectrogram_num_cols, spectrograms[i].cols());
  }
}

TEST_F(SpectrogramCalculatorTest, MultichannelSpectrogramValuesAreRight) {
  const std::vector<int> input_packet_sizes = {
      460};  // 100 + 9*40 for 10 frames.
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_allow_multichannel_input(true);
  num_input_channels_ = 10;
  InitializeGraph();
  FillInputHeader();
  const float tone_frequency_hz = 440.0;
  SetupMultichannelInputPackets(input_packet_sizes, tone_frequency_hz);

  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  auto spectrograms = output().packets[0].Get<std::vector<Matrix>>();
  int num_output_frames = spectrograms[0].cols();
  for (int i = 0; i < num_input_channels_; i++) {
    for (int frame = 0; frame < num_output_frames; ++frame) {
      if (i % 2 == 0) {
        CheckPeakFrequencyInMatrix(spectrograms[i], frame, 0);
      } else {
        CheckPeakFrequencyInMatrix(spectrograms[i], frame, tone_frequency_hz);
      }
    }
  }
}

TEST_F(SpectrogramCalculatorTest, MultichannelHandlesShortInitialPacket) {
  // First packet is less than one frame, but second packet should trigger a
  // complete frame from all channels.
  const std::vector<int> input_packet_sizes = {50, 50};
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  options_.set_allow_multichannel_input(true);
  num_input_channels_ = 2;
  InitializeGraph();
  FillInputHeader();
  const float tone_frequency_hz = 440.0;
  SetupCosineInputPackets(input_packet_sizes, tone_frequency_hz);
  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  auto spectrograms = output().packets[0].Get<std::vector<Matrix>>();
  EXPECT_FLOAT_EQ(spectrograms.size(), num_input_channels_);
  int spectrogram_num_rows = spectrograms[0].rows();
  int spectrogram_num_cols = spectrograms[0].cols();
  for (int i = 1; i < num_input_channels_; i++) {
    EXPECT_EQ(spectrogram_num_rows, spectrograms[i].rows());
    EXPECT_EQ(spectrogram_num_cols, spectrograms[i].cols());
  }
}

TEST_F(SpectrogramCalculatorTest,
       MultichannelComplexHandlesShortInitialPacket) {
  // First packet is less than one frame, but second packet should trigger a
  // complete frame from all channels, even for complex output.
  const std::vector<int> input_packet_sizes = {50, 50};
  options_.set_frame_duration_seconds(100.0 / input_sample_rate_);
  options_.set_frame_overlap_seconds(60.0 / input_sample_rate_);
  options_.set_pad_final_packet(false);
  options_.set_allow_multichannel_input(true);
  options_.set_output_type(SpectrogramCalculatorOptions::COMPLEX);
  num_input_channels_ = 2;
  InitializeGraph();
  FillInputHeader();
  const float tone_frequency_hz = 440.0;
  SetupCosineInputPackets(input_packet_sizes, tone_frequency_hz);
  MP_ASSERT_OK(Run());

  CheckOutputHeadersAndTimestamps();
  auto spectrograms = output().packets[0].Get<std::vector<Eigen::MatrixXcf>>();
  EXPECT_FLOAT_EQ(spectrograms.size(), num_input_channels_);
  int spectrogram_num_rows = spectrograms[0].rows();
  int spectrogram_num_cols = spectrograms[0].cols();
  for (int i = 1; i < num_input_channels_; i++) {
    EXPECT_EQ(spectrogram_num_rows, spectrograms[i].rows());
    EXPECT_EQ(spectrogram_num_cols, spectrograms[i].cols());
  }
}

void BM_ProcessDC(benchmark::State& state) {
  CalculatorGraphConfig::Node node_config;
  node_config.set_calculator("SpectrogramCalculator");
  node_config.add_input_stream("input_audio");
  node_config.add_output_stream("output_spectrogram");

  SpectrogramCalculatorOptions* options =
      node_config.mutable_options()->MutableExtension(
          SpectrogramCalculatorOptions::ext);
  options->set_frame_duration_seconds(0.010);
  options->set_frame_overlap_seconds(0.0);
  options->set_pad_final_packet(false);
  *node_config.mutable_options()->MutableExtension(
      SpectrogramCalculatorOptions::ext) = *options;

  int num_input_channels = 1;
  int packet_size_samples = 1600000;
  TimeSeriesHeader* header = new TimeSeriesHeader();
  header->set_sample_rate(16000.0);
  header->set_num_channels(num_input_channels);

  CalculatorRunner runner(node_config);
  runner.MutableInputs()->Index(0).header = Adopt(header);

  Matrix* payload = new Matrix(
      Matrix::Constant(num_input_channels, packet_size_samples, 1.0));
  Timestamp timestamp = Timestamp(0);
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(payload).At(timestamp));

  for (auto _ : state) {
    ASSERT_TRUE(runner.Run().ok());
  }

  const CalculatorRunner::StreamContents& output = runner.Outputs().Index(0);
  const Matrix& output_matrix = output.packets[0].Get<Matrix>();
  LOG(INFO) << "Output matrix=" << output_matrix.rows() << "x"
            << output_matrix.cols();
  LOG(INFO) << "First values=" << output_matrix(0, 0) << ", "
            << output_matrix(1, 0) << ", " << output_matrix(2, 0) << ", "
            << output_matrix(3, 0);
}

BENCHMARK(BM_ProcessDC);

}  // anonymous namespace
}  // namespace mediapipe
