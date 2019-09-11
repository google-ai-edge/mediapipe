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
#include <vector>

#include "Eigen/Core"
#include "mediapipe/calculators/audio/mfcc_mel_calculators.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {

// Use a sample rate that is unlikely to be a default somewhere.
const float kAudioSampleRate = 8800.0;

template <typename OptionsType, const char* CalculatorName>
class FramewiseTransformCalculatorTest
    : public TimeSeriesCalculatorTest<OptionsType> {
 protected:
  void SetUp() override {
    this->calculator_name_ = CalculatorName;
    this->num_input_channels_ = 129;
    // This is the frame rate coming out of the SpectrogramCalculator.
    this->input_sample_rate_ = 100.0;
  }

  // Returns the number of samples per packet.
  int GenerateRandomNonnegInputStream(int num_packets) {
    const double kSecondsPerPacket = 0.2;
    const int num_samples_per_packet =
        kSecondsPerPacket * this->input_sample_rate_;
    for (int i = 0; i < num_packets; ++i) {
      const int timestamp =
          i * kSecondsPerPacket * Timestamp::kTimestampUnitsPerSecond;
      // Mfcc, MelSpectrum expect squared-magnitude inputs, so make
      // sure the input data has no negative values.
      Matrix* sqdata = this->NewRandomMatrix(this->num_input_channels_,
                                             num_samples_per_packet);
      *sqdata = sqdata->array().square();
      this->AppendInputPacket(sqdata, timestamp);
    }
    return num_samples_per_packet;
  }

  void CheckOutputPacketMetadata(int expected_num_channels,
                                 int expected_num_samples_per_packet) {
    int expected_timestamp = 0;
    for (const auto& packet : this->output().packets) {
      EXPECT_EQ(expected_timestamp, packet.Timestamp().Value());
      expected_timestamp += expected_num_samples_per_packet /
                            this->input_sample_rate_ *
                            Timestamp::kTimestampUnitsPerSecond;

      const Matrix& output_matrix = packet.template Get<Matrix>();

      EXPECT_EQ(output_matrix.rows(), expected_num_channels);
      EXPECT_EQ(output_matrix.cols(), expected_num_samples_per_packet);
    }
  }

  void SetupGraphAndHeader() {
    this->InitializeGraph();
    this->FillInputHeader();
  }

  // Argument is the expected number of dimensions (channels, columns) in
  // the output data from the Calculator under test, which the test should
  // know.
  void SetupRandomInputPackets() {
    constexpr int kNumPackets = 5;
    num_samples_per_packet_ = GenerateRandomNonnegInputStream(kNumPackets);
  }

  ::mediapipe::Status Run() { return this->RunGraph(); }

  void CheckResults(int expected_num_channels) {
    const auto& output_header =
        this->output().header.template Get<TimeSeriesHeader>();
    EXPECT_EQ(this->input_sample_rate_, output_header.sample_rate());
    CheckOutputPacketMetadata(expected_num_channels, num_samples_per_packet_);

    // Sanity check that output packets have non-zero energy.
    for (const auto& packet : this->output().packets) {
      const Matrix& data = packet.template Get<Matrix>();
      EXPECT_GT(data.squaredNorm(), 0);
    }
  }

  // Allows SetupRandomInputPackets() to inform CheckResults() about how
  // big the packets are supposed to be.
  int num_samples_per_packet_;
};

constexpr char kMfccCalculator[] = "MfccCalculator";
typedef FramewiseTransformCalculatorTest<MfccCalculatorOptions, kMfccCalculator>
    MfccCalculatorTest;
TEST_F(MfccCalculatorTest, AudioSampleRateFromInputHeader) {
  audio_sample_rate_ = kAudioSampleRate;
  SetupGraphAndHeader();
  SetupRandomInputPackets();

  MP_EXPECT_OK(Run());

  CheckResults(options_.mfcc_count());
}
TEST_F(MfccCalculatorTest, NoAudioSampleRate) {
  // Leave audio_sample_rate_ == kUnset, so it is not present in the
  // input TimeSeriesHeader; expect failure.
  SetupGraphAndHeader();
  SetupRandomInputPackets();

  EXPECT_FALSE(Run().ok());
}

constexpr char kMelSpectrumCalculator[] = "MelSpectrumCalculator";
typedef FramewiseTransformCalculatorTest<MelSpectrumCalculatorOptions,
                                         kMelSpectrumCalculator>
    MelSpectrumCalculatorTest;
TEST_F(MelSpectrumCalculatorTest, AudioSampleRateFromInputHeader) {
  audio_sample_rate_ = kAudioSampleRate;
  SetupGraphAndHeader();
  SetupRandomInputPackets();

  MP_EXPECT_OK(Run());

  CheckResults(options_.channel_count());
}
TEST_F(MelSpectrumCalculatorTest, NoAudioSampleRate) {
  // Leave audio_sample_rate_ == kUnset, so it is not present in the
  // input TimeSeriesHeader; expect failure.
  SetupGraphAndHeader();
  SetupRandomInputPackets();

  EXPECT_FALSE(Run().ok());
}
}  // namespace mediapipe
