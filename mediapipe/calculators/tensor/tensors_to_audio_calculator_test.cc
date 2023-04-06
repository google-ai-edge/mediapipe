// Copyright 2022 The MediaPipe Authors.
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

#include <algorithm>
#include <new>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/audio_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_audio_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using Options = ::mediapipe::TensorsToAudioCalculatorOptions;

class TensorsToAudioCalculatorFftTest : public ::testing::Test {
 protected:
  // Creates an audio matrix containing a single sample of 1.0 at a specified
  // offset.
  Matrix CreateImpulseSignalData(int64_t num_samples, int impulse_offset_idx) {
    Matrix impulse = Matrix::Zero(1, num_samples);
    impulse(0, impulse_offset_idx) = 1.0;
    return impulse;
  }

  void ConfigGraph(int num_samples, double sample_rate, int fft_size,
                   Options::DftTensorFormat dft_tensor_format) {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
        R"(
        input_stream: "audio_in"
        input_stream: "sample_rate"
        output_stream: "audio_out"
        node {
          calculator: "AudioToTensorCalculator"
          input_stream: "AUDIO:audio_in"
          input_stream: "SAMPLE_RATE:sample_rate"
          output_stream: "TENSORS:tensors"
          output_stream: "DC_AND_NYQUIST:dc_and_nyquist"
          options {
            [mediapipe.AudioToTensorCalculatorOptions.ext] {
              num_channels: 1
              num_samples: $0
              num_overlapping_samples: 0
              target_sample_rate: $1
              fft_size: $2
              dft_tensor_format: $3
            }
          }
        }
        node {
          calculator: "TensorsToAudioCalculator"
          input_stream: "TENSORS:tensors"
          input_stream: "DC_AND_NYQUIST:dc_and_nyquist"
          output_stream: "AUDIO:audio_out"
          options {
            [mediapipe.TensorsToAudioCalculatorOptions.ext] {
              fft_size: $2
              dft_tensor_format: $3
            }
          }
        }
        )",
        /*$0=*/num_samples,
        /*$1=*/sample_rate,
        /*$2=*/fft_size,
        /*$3=*/Options::DftTensorFormat_Name(dft_tensor_format)));
    tool::AddVectorSink("audio_out", &graph_config_, &audio_out_packets_);
  }

  void RunGraph(const Matrix& input_data, double sample_rate) {
    MP_ASSERT_OK(graph_.Initialize(graph_config_));
    MP_ASSERT_OK(graph_.StartRun({}));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "sample_rate", MakePacket<double>(sample_rate).At(Timestamp(0))));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "audio_in", MakePacket<Matrix>(input_data).At(Timestamp(0))));
    MP_ASSERT_OK(graph_.CloseAllInputStreams());
    MP_ASSERT_OK(graph_.WaitUntilDone());
  }

  std::vector<Packet> audio_out_packets_;
  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
};

TEST_F(TensorsToAudioCalculatorFftTest, TestInvalidFftSize) {
  ConfigGraph(320, 16000, 103, Options::WITH_NYQUIST);
  MP_ASSERT_OK(graph_.Initialize(graph_config_));
  MP_ASSERT_OK(graph_.StartRun({}));
  auto status = graph_.WaitUntilIdle();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("FFT size must be of the form"));
}

TEST_F(TensorsToAudioCalculatorFftTest, TestImpulseSignalAtTheCenter) {
  constexpr int sample_size = 320;
  constexpr double sample_rate = 16000;
  ConfigGraph(sample_size, sample_rate, 320, Options::WITH_NYQUIST);
  Matrix impulse_data = CreateImpulseSignalData(sample_size, sample_size / 2);
  RunGraph(impulse_data, sample_rate);
  ASSERT_EQ(1, audio_out_packets_.size());
  MP_ASSERT_OK(audio_out_packets_[0].ValidateAsType<Matrix>());
  // The impulse signal at the center is not affected by the window function.
  EXPECT_EQ(audio_out_packets_[0].Get<Matrix>(), impulse_data);
}

TEST_F(TensorsToAudioCalculatorFftTest, TestWindowedImpulseSignal) {
  constexpr int sample_size = 320;
  constexpr double sample_rate = 16000;
  ConfigGraph(sample_size, sample_rate, 320, Options::WITH_NYQUIST);
  Matrix impulse_data = CreateImpulseSignalData(sample_size, sample_size / 4);
  RunGraph(impulse_data, sample_rate);
  ASSERT_EQ(1, audio_out_packets_.size());
  MP_ASSERT_OK(audio_out_packets_[0].ValidateAsType<Matrix>());
  // As the impulse signal sits at the 1/4 of the hann window, the inverse
  // window function reduces it by half.
  EXPECT_EQ(audio_out_packets_[0].Get<Matrix>(), impulse_data / 2);
}

TEST_F(TensorsToAudioCalculatorFftTest, TestImpulseSignalAtBeginning) {
  constexpr int sample_size = 320;
  constexpr double sample_rate = 16000;
  ConfigGraph(sample_size, sample_rate, 320, Options::WITH_NYQUIST);
  Matrix impulse_data = CreateImpulseSignalData(sample_size, 0);
  RunGraph(impulse_data, sample_rate);
  ASSERT_EQ(1, audio_out_packets_.size());
  MP_ASSERT_OK(audio_out_packets_[0].ValidateAsType<Matrix>());
  // As the impulse signal sits at the beginning of the hann window, the inverse
  // window function completely removes it.
  EXPECT_EQ(audio_out_packets_[0].Get<Matrix>(), Matrix::Zero(1, sample_size));
}

TEST_F(TensorsToAudioCalculatorFftTest, TestDftTensorWithDCAndNyquist) {
  constexpr int sample_size = 320;
  constexpr double sample_rate = 16000;
  ConfigGraph(sample_size, sample_rate, 320, Options::WITH_DC_AND_NYQUIST);

  Matrix impulse_data = CreateImpulseSignalData(sample_size, sample_size / 2);
  RunGraph(impulse_data, sample_rate);
  ASSERT_EQ(1, audio_out_packets_.size());
  MP_ASSERT_OK(audio_out_packets_[0].ValidateAsType<Matrix>());
  // The impulse signal at the center is not affected by the window function.
  EXPECT_EQ(audio_out_packets_[0].Get<Matrix>(), impulse_data);
}

TEST_F(TensorsToAudioCalculatorFftTest, TestDftTensorWithoutDCAndNyquist) {
  constexpr int sample_size = 320;
  constexpr double sample_rate = 16000;
  ConfigGraph(sample_size, sample_rate, 320, Options::WITHOUT_DC_AND_NYQUIST);

  Matrix impulse_data = CreateImpulseSignalData(sample_size, sample_size / 2);
  RunGraph(impulse_data, sample_rate);
  ASSERT_EQ(1, audio_out_packets_.size());
  MP_ASSERT_OK(audio_out_packets_[0].ValidateAsType<Matrix>());
  // The impulse signal at the center is not affected by the window function.
  EXPECT_EQ(audio_out_packets_[0].Get<Matrix>(), impulse_data);
}

}  // namespace
}  // namespace mediapipe
