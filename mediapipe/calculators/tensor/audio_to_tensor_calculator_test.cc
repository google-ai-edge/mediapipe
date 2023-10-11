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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "audio/dsp/resampler_q.h"
#include "mediapipe/calculators/tensor/audio_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

using ::testing::Not;
using Options = ::mediapipe::AudioToTensorCalculatorOptions;
using FlushMode = Options::FlushMode;

int DivideRoundedUp(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

std::unique_ptr<Matrix> CreateTestMatrix(int num_channels, int num_samples,
                                         int timestamp) {
  auto matrix = std::make_unique<Matrix>(num_channels, num_samples);
  for (int c = 0; c < num_channels; ++c) {
    for (int i = 0; i < num_samples; ++i) {
      // A float value with the sample, channel, and timestamp separated by a
      // few orders of magnitude, for easy parsing by humans.
      (*matrix)(c, i) = timestamp / 10000 + i + c / 100.0;
    }
  }
  return matrix;
}

std::unique_ptr<Matrix> ResampleBuffer(const Matrix& input_matrix,
                                       double resampling_factor) {
  audio_dsp::QResamplerParams params;
  std::vector<float> resampled;
  int num_channels = input_matrix.rows();
  std::vector<float> input_data(input_matrix.data(),
                                input_matrix.data() + input_matrix.size());
  resampled = audio_dsp::QResampleSignal<float>(
      1, resampling_factor, num_channels, params, input_data);
  Matrix res = Eigen::Map<Matrix>(resampled.data(), num_channels,
                                  resampled.size() / num_channels);
  return std::make_unique<Matrix>(std::move(res));
}

class AudioToTensorCalculatorNonStreamingModeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void Run(int num_samples, int num_overlapping_samples,
           double resampling_factor, const Matrix& input_matrix,
           int num_channels_override = 0) {
    const int num_channels = num_channels_override == 0 ? input_matrix.rows()
                                                        : num_channels_override;
    double input_sample_rate = 10000;
    double target_sample_rate = input_sample_rate * resampling_factor;
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        absl::Substitute(R"(
        input_stream: "audio"
        input_stream: "sample_rate"
        output_stream: "tensors"
        output_stream: "timestamps"
        node {
          calculator: "AudioToTensorCalculator"
          input_stream: "AUDIO:audio"
          input_stream: "SAMPLE_RATE:sample_rate"
          output_stream: "TENSORS:tensors"
          output_stream: "TIMESTAMPS:timestamps"
          options {
            [mediapipe.AudioToTensorCalculatorOptions.ext] {
              num_channels: $0
              num_samples: $1
              num_overlapping_samples: $2
              target_sample_rate: $3
              stream_mode: false
            }
          }
        }
        )",
                         /*$0=*/num_channels,
                         /*$1=*/num_samples, /*$2=*/num_overlapping_samples,
                         /*$3=*/target_sample_rate));
    tool::AddVectorSink("tensors", &graph_config, &tensors_packets_);
    tool::AddVectorSink("timestamps", &graph_config, &timestamps_packets_);

    // Run the graph.
    MP_ASSERT_OK(graph_.Initialize(graph_config));
    MP_ASSERT_OK(graph_.StartRun({}));
    // Run with the input matrix multiple times.
    for (int i = 0; i < num_iterations_; ++i) {
      MP_ASSERT_OK(graph_.AddPacketToInputStream(
          "audio",
          MakePacket<Matrix>(input_matrix)
              .At(Timestamp(i * Timestamp::kTimestampUnitsPerSecond))));
      MP_ASSERT_OK(graph_.AddPacketToInputStream(
          "sample_rate",
          MakePacket<double>(input_sample_rate)
              .At(Timestamp(i * Timestamp::kTimestampUnitsPerSecond))));
    }
    MP_ASSERT_OK(graph_.CloseAllInputStreams());
    MP_ASSERT_OK(graph_.WaitUntilIdle());
  }

  void CheckTensorsOutputPackets(const Matrix& expected_matrix,
                                 int sample_offset, int num_tensors_per_input,
                                 bool mono = false) {
    ASSERT_EQ(num_iterations_ * num_tensors_per_input, tensors_packets_.size());
    for (int i = 0; i < num_iterations_; ++i) {
      for (int j = 0; j < num_tensors_per_input; ++j) {
        CheckTensorsOutputPacket(
            expected_matrix, tensors_packets_[i * num_tensors_per_input + j],
            /*sample_offset=*/sample_offset * j, /*index=*/j, /*mono=*/mono);
      }
    }
  }

  void CheckTensorsOutputPacket(const Matrix& expected_matrix,
                                const Packet& packet, int sample_offset,
                                int index, bool mono = false) {
    MP_ASSERT_OK(packet.ValidateAsType<std::vector<Tensor>>());
    ASSERT_EQ(1, packet.Get<std::vector<Tensor>>().size());
    const Tensor& output_tensor = packet.Get<std::vector<Tensor>>()[0];
    auto* buffer = output_tensor.GetCpuReadView().buffer<float>();
    int num_values = output_tensor.shape().num_elements();
    const std::vector<float> output_floats(buffer, buffer + num_values);
    for (int i = 0; i < num_values; ++i) {
      if (i + sample_offset >= expected_matrix.size()) {
        EXPECT_FLOAT_EQ(output_floats[i], 0);
      } else if (mono) {
        EXPECT_FLOAT_EQ(output_floats[i],
                        expected_matrix.coeff(0, i + sample_offset));
      } else {
        // Stereo.
        EXPECT_FLOAT_EQ(output_floats[i],
                        expected_matrix.coeff((i + sample_offset) % 2,
                                              (i + sample_offset) / 2))
            << "i=" << i << ", sample_offset=" << sample_offset;
      }
    }
  }

  void CheckTimestampsOutputPackets(
      std::vector<int64_t> expected_timestamp_values) {
    ASSERT_EQ(num_iterations_, timestamps_packets_.size());
    for (int i = 0; i < timestamps_packets_.size(); ++i) {
      const auto& p = timestamps_packets_[i];
      MP_ASSERT_OK(p.ValidateAsType<std::vector<Timestamp>>());
      auto output_timestamps = p.Get<std::vector<Timestamp>>();
      int64_t base_timestamp = i * Timestamp::kTimestampUnitsPerSecond;
      std::vector<Timestamp> expected_timestamps;
      expected_timestamps.resize(expected_timestamp_values.size());
      std::transform(expected_timestamp_values.begin(),
                     expected_timestamp_values.end(),
                     expected_timestamps.begin(),
                     [base_timestamp](int64_t v) -> Timestamp {
                       return Timestamp(v + base_timestamp);
                     });
      EXPECT_EQ(expected_timestamps, output_timestamps);
      EXPECT_EQ(p.Timestamp(), expected_timestamps.back());
    }
  }

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  void CloseGraph() { MP_EXPECT_OK(graph_.WaitUntilDone()); }

 private:
  CalculatorGraph graph_;
  int num_iterations_ = 10;
  std::vector<Packet> tensors_packets_;
  std::vector<Packet> timestamps_packets_;
};

TEST_F(AudioToTensorCalculatorNonStreamingModeTest,
       ConvertToNoOverlappingFp32Tensors) {
  auto input_matrix = CreateTestMatrix(2, 8, 0);
  Run(/*num_samples=*/4, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/1.0f, *input_matrix);
  CheckTensorsOutputPackets(*input_matrix, /*sample_offset=*/8,
                            /*num_tensors_per_input=*/2);
  CheckTimestampsOutputPackets({0, 400});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest,
       ConvertToOverlappingFp32Tensors) {
  auto input_matrix = CreateTestMatrix(2, 8, 0);
  Run(/*num_samples=*/4, /*num_overlapping_samples=*/2,
      /*resampling_factor=*/1.0f, *input_matrix);
  CheckTensorsOutputPackets(*input_matrix, /*sample_offset=*/4,
                            /*num_tensors_per_input=*/4);
  CheckTimestampsOutputPackets({0, 200, 400, 600});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest, TensorsWithZeroPadding) {
  auto input_matrix = CreateTestMatrix(2, 7, 0);
  Run(/*num_samples=*/4, /*num_overlapping_samples=*/2,
      /*resampling_factor=*/1.0f, *input_matrix);
  CheckTensorsOutputPackets(*input_matrix, /*sample_offset=*/4,
                            /*num_tensors_per_input=*/3);
  CheckTimestampsOutputPackets({0, 200, 400});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest, Mixdown) {
  auto input_matrix = CreateTestMatrix(2, 8, 0);
  Run(/*num_samples=*/4, /*num_overlapping_samples=*/2,
      /*resampling_factor=*/1.0f, *input_matrix, /*num_channels_override=*/1);
  const Matrix& mono_matrix = input_matrix->colwise().mean();
  CheckTensorsOutputPackets(mono_matrix, /*sample_offset=*/2,
                            /*num_tensors_per_input=*/4, /*mono=*/true);
  CheckTimestampsOutputPackets({0, 200, 400, 600});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest, Downsampling) {
  auto input_matrix = CreateTestMatrix(2, 1024, 0);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/0.5f, *input_matrix);
  auto expected_matrix =
      ResampleBuffer(*input_matrix, /*resampling_factor=*/0.5f);
  CheckTensorsOutputPackets(*expected_matrix, /*sample_offset=*/512,
                            /*num_tensors_per_input=*/3);
  CheckTimestampsOutputPackets({0, 51200, 102400});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest,
       DownsamplingWithOverlapping) {
  auto input_matrix = CreateTestMatrix(2, 1024, 0);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/64,
      /*resampling_factor=*/0.5f, *input_matrix);
  auto expected_matrix =
      ResampleBuffer(*input_matrix, /*resampling_factor=*/0.5f);
  CheckTensorsOutputPackets(*expected_matrix, /*sample_offset=*/384,
                            /*num_tensors_per_input=*/3);
  CheckTimestampsOutputPackets({0, 38400, 76800});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest, Upsampling) {
  auto input_matrix = CreateTestMatrix(2, 1024, 0);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/2.0f, *input_matrix);
  auto expected_matrix =
      ResampleBuffer(*input_matrix, /*resampling_factor=*/2.0f);
  CheckTensorsOutputPackets(*expected_matrix,
                            /*sample_offset=*/512,
                            /*num_tensors_per_input=*/9);
  CheckTimestampsOutputPackets(
      {0, 12800, 25600, 38400, 51200, 64000, 76800, 89600, 102400});
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorNonStreamingModeTest, UpsamplingWithOverlapping) {
  auto input_matrix = CreateTestMatrix(2, 256, 0);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/64,
      /*resampling_factor=*/2.0f, *input_matrix);
  auto expected_matrix =
      ResampleBuffer(*input_matrix, /*resampling_factor=*/2.0f);
  CheckTensorsOutputPackets(*expected_matrix,
                            /*sample_offset=*/384,
                            /*num_tensors_per_input=*/3);
  CheckTimestampsOutputPackets({0, 9600, 19200});
  CloseGraph();
}

class AudioToTensorCalculatorStreamingModeTest : public ::testing::Test {
 protected:
  void SetUp() override { sample_buffer_ = std::make_unique<Matrix>(2, 0); }

  void SetInputBufferNumSamplesPerChannel(int num_samples) {
    input_buffer_num_samples_ = num_samples;
  }

  void SetNumIterations(int num_iterations) {
    num_iterations_ = num_iterations;
  }

  int GetExpectedNumOfSamples() { return output_sample_buffer_->cols(); }

  void Run(int num_samples, int num_overlapping_samples,
           double resampling_factor, int padding_before = 0,
           int padding_after = 0, bool expect_init_error = false) {
    double input_sample_rate = 10000;
    double target_sample_rate = input_sample_rate * resampling_factor;
    FlushMode flush_mode = (padding_before != 0 || padding_after != 0)
                               ? Options::PROCEED_AS_USUAL
                               : Options::ENTIRE_TAIL_AT_TIMESTAMP_MAX;

    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        absl::Substitute(R"(
        input_stream: "audio"
        input_stream: "sample_rate"
        output_stream: "tensors"
        node {
          calculator: "AudioToTensorCalculator"
          input_stream: "AUDIO:audio"
          input_stream: "SAMPLE_RATE:sample_rate"
          output_stream: "TENSORS:tensors"
          options {
            [mediapipe.AudioToTensorCalculatorOptions.ext] {
              num_channels: 2
              num_samples: $0
              num_overlapping_samples: $1
              target_sample_rate: $2
              stream_mode:true
              padding_samples_before: $3
              padding_samples_after: $4
              flush_mode: $5
            }
          }
        }
        )",
                         /*$0=*/num_samples, /*$1=*/num_overlapping_samples,
                         /*$2=*/target_sample_rate, /*$3=*/padding_before,
                         /*$4=*/padding_after, /*$5=*/flush_mode));
    tool::AddVectorSink("tensors", &graph_config, &tensors_packets_);

    // Run the graph.
    const absl::Status init_status = graph_.Initialize(graph_config);
    if (expect_init_error) {
      EXPECT_THAT(init_status, Not(IsOk()));
      return;
    }
    MP_ASSERT_OK(init_status);
    MP_ASSERT_OK(graph_.StartRun({}));
    for (int i = 0; i < num_iterations_; ++i) {
      Timestamp input_timestamp(Timestamp::kTimestampUnitsPerSecond * i);
      auto new_data = CreateTestMatrix(2, input_buffer_num_samples_,
                                       input_timestamp.Value());
      MP_ASSERT_OK(graph_.AddPacketToInputStream(
          "audio", MakePacket<Matrix>(*new_data).At(input_timestamp)));
      MP_ASSERT_OK(graph_.AddPacketToInputStream(
          "sample_rate",
          MakePacket<double>(input_sample_rate).At(input_timestamp)));
      sample_buffer_->conservativeResize(
          Eigen::NoChange, sample_buffer_->cols() + new_data->cols());
      sample_buffer_->rightCols(new_data->cols()).swap(*new_data);
    }
    MP_ASSERT_OK(graph_.CloseAllInputStreams());
    MP_ASSERT_OK(graph_.WaitUntilIdle());
    if (resampling_factor == 1) {
      output_sample_buffer_ = std::make_unique<Matrix>(*sample_buffer_);
    } else {
      output_sample_buffer_ =
          ResampleBuffer(*sample_buffer_, resampling_factor);
    }
    if (padding_before != 0 || padding_after != 0) {
      Matrix padded = Matrix::Zero(
          2, padding_before + output_sample_buffer_->cols() + padding_after);
      padded.block(0, padding_before, 2, output_sample_buffer_->cols()) =
          *output_sample_buffer_;
      output_sample_buffer_->swap(padded);
    }
  }

  void CheckTensorsOutputPackets(int sample_offset, int num_packets,
                                 int64_t timestamp_interval,
                                 bool output_last_at_close) {
    ASSERT_EQ(num_packets, tensors_packets_.size());
    for (int i = 0; i < num_packets; ++i) {
      if (i == num_packets - 1 && output_last_at_close) {
        CheckTensorsOutputPacket(sample_offset * i, i, Timestamp::Max());
      } else {
        CheckTensorsOutputPacket(sample_offset * i, i,
                                 Timestamp(timestamp_interval * i));
      }
    }
  }

  void CheckTensorsOutputPacket(int sample_offset, int index,
                                Timestamp expected_timestamp) {
    const Packet& p = tensors_packets_[index];
    MP_ASSERT_OK(p.ValidateAsType<std::vector<Tensor>>());
    const Tensor& output_tensor = p.Get<std::vector<Tensor>>()[0];
    auto buffer = output_tensor.GetCpuReadView().buffer<float>();
    int num_values = output_tensor.shape().num_elements();
    std::vector<float> output_floats(buffer, buffer + num_values);
    for (int i = 0; i < num_values; ++i) {
      if (i + sample_offset >= output_sample_buffer_->size()) {
        EXPECT_FLOAT_EQ(output_floats[i], 0);
      } else {
        EXPECT_NEAR(output_floats[i],
                    output_sample_buffer_->coeff((i + sample_offset) % 2,
                                                 (i + sample_offset) / 2),
                    0.001)
            << "i=" << i << ", sample_offset=" << sample_offset
            << ", packet index=" << index;
      }
    }
    EXPECT_EQ(p.Timestamp(), expected_timestamp);
  }

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  absl::Status TryCloseGraph() { return graph_.WaitUntilDone(); }
  void CloseGraph() { MP_EXPECT_OK(TryCloseGraph()); }

 private:
  int input_buffer_num_samples_ = 10;
  int num_iterations_ = 10;
  CalculatorGraph graph_;
  std::vector<Packet> tensors_packets_;
  std::unique_ptr<Matrix> sample_buffer_;
  std::unique_ptr<Matrix> output_sample_buffer_;
};

TEST_F(AudioToTensorCalculatorStreamingModeTest,
       OutputNoOverlappingFp32Tensors) {
  Run(/*num_samples=*/5, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/1.0f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/10,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 5),
      /*timestamp_interval=*/500,
      /*output_last_at_close=*/false);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, OutputRemainingInCloseMethod) {
  Run(/*num_samples=*/6, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/1.0f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/12,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 6),
      /*timestamp_interval=*/600,
      /*output_last_at_close=*/true);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, OutputOverlappingFp32Tensors) {
  SetInputBufferNumSamplesPerChannel(12);
  Run(/*num_samples=*/10, /*num_overlapping_samples=*/2,
      /*resampling_factor=*/1.0f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/16,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 8),
      /*timestamp_interval=*/800,
      /*output_last_at_close=*/true);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, Downsampling) {
  SetInputBufferNumSamplesPerChannel(1000);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/0.5f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/512,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 256),
      /*timestamp_interval=*/51200,
      /*output_last_at_close=*/true);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, DownsamplingWithOverlapping) {
  SetInputBufferNumSamplesPerChannel(1024);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/64,
      /*resampling_factor=*/0.5f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/384,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 192),
      /*timestamp_interval=*/38400,
      /*output_last_at_close=*/true);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, Upsampling) {
  SetInputBufferNumSamplesPerChannel(1000);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/2.0f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/512,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 256),
      /*timestamp_interval=*/12800,
      /*output_last_at_close=*/true);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, UpsamplingWithOverlapping) {
  SetInputBufferNumSamplesPerChannel(1024);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/64,
      /*resampling_factor=*/2.0f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/384,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 192),
      /*timestamp_interval=*/9600,
      /*output_last_at_close=*/true);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest,
       UpsamplingWithOverlappingAndPadding) {
  SetInputBufferNumSamplesPerChannel(1024);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/64,
      /*resampling_factor=*/2.0f, /*padding_before=*/13, /*padding_after=*/999);
  CheckTensorsOutputPackets(
      /*sample_offset=*/384,
      /*num_packets=*/DivideRoundedUp(GetExpectedNumOfSamples(), 192),
      /*timestamp_interval=*/9600,
      /*output_last_at_close=*/false);
  CloseGraph();
}

TEST_F(AudioToTensorCalculatorStreamingModeTest, NegativePaddingUnsupported) {
  SetInputBufferNumSamplesPerChannel(1024);
  Run(/*num_samples=*/256, /*num_overlapping_samples=*/64,
      /*resampling_factor=*/2.0f, /*padding_before=*/13, /*padding_after=*/-3,
      /*expect_init_error=*/true);
  EXPECT_THAT(TryCloseGraph(), Not(IsOk()));
}

TEST_F(AudioToTensorCalculatorStreamingModeTest,
       OnlyOutputInCloseIfNoSufficientSamples) {
  SetNumIterations(1);
  Run(/*num_samples=*/8, /*num_overlapping_samples=*/0,
      /*resampling_factor=*/0.5f);
  CheckTensorsOutputPackets(
      /*sample_offset=*/0,
      /*num_packets=*/1,
      /*timestamp_interval=*/0,
      /*output_last_at_close=*/true);
  CloseGraph();
}

class AudioToTensorCalculatorFftTest : public ::testing::Test {
 protected:
  // Creates an audio matrix containing a single sample of 1.0 at a specified
  // offset.
  std::unique_ptr<Matrix> CreateImpulseSignalData(int64_t num_samples,
                                                  int impulse_offset_idx) {
    Matrix impulse = Matrix::Zero(1, num_samples);
    impulse(0, impulse_offset_idx) = 1.0;
    return std::make_unique<Matrix>(std::move(impulse));
  }

  void ConfigGraph(int num_channels, int num_samples,
                   int num_overlapping_samples, double sample_rate,
                   int fft_size) {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(
        absl::Substitute(R"(
        input_stream: "audio"
        input_stream: "sample_rate"
        output_stream: "tensors"
        output_stream: "dc_and_nyquist"
        node {
          calculator: "AudioToTensorCalculator"
          input_stream: "AUDIO:audio"
          input_stream: "SAMPLE_RATE:sample_rate"
          output_stream: "TENSORS:tensors"
          output_stream: "DC_AND_NYQUIST:dc_and_nyquist"
          options {
            [mediapipe.AudioToTensorCalculatorOptions.ext] {
              num_channels: $0
              num_samples: $1
              num_overlapping_samples: $2
              target_sample_rate: $3
              fft_size: $4
            }
          }
        }
        )",
                         /*$0=*/num_channels,
                         /*$1=*/num_samples,
                         /*$2=*/num_overlapping_samples,
                         /*$3=*/sample_rate, /*$4=*/fft_size));
    std::vector<Packet> tensors_packets;
    tool::AddVectorSink("tensors", &graph_config_, &tensors_packets_);
    std::vector<Packet> dc_and_nyquist_packets;
    tool::AddVectorSink("dc_and_nyquist", &graph_config_,
                        &dc_and_nyquist_packets_);
  }

  void RunGraph(std::unique_ptr<Matrix> input_data, double sample_rate) {
    MP_ASSERT_OK(graph_.Initialize(graph_config_));
    MP_ASSERT_OK(graph_.StartRun({}));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "sample_rate", MakePacket<double>(sample_rate).At(Timestamp(0))));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "audio", MakePacket<Matrix>(*input_data).At(Timestamp(0))));
    MP_ASSERT_OK(graph_.CloseAllInputStreams());
    MP_ASSERT_OK(graph_.WaitUntilIdle());
    ASSERT_EQ(tensors_packets_.size(), dc_and_nyquist_packets_.size());
  }

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  void CloseGraph() { MP_EXPECT_OK(graph_.WaitUntilDone()); }

  std::vector<Packet> tensors_packets_;
  std::vector<Packet> dc_and_nyquist_packets_;
  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
};

TEST_F(AudioToTensorCalculatorFftTest, TestInvalidFftSize) {
  ConfigGraph(1, 320, 160, 16000, 103);
  MP_ASSERT_OK(graph_.Initialize(graph_config_));
  MP_ASSERT_OK(graph_.StartRun({}));
  auto status = graph_.WaitUntilIdle();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("FFT size must be of the form"));
}

TEST_F(AudioToTensorCalculatorFftTest, TestInvalidNumChannels) {
  ConfigGraph(3, 320, 160, 16000, 256);
  MP_ASSERT_OK(graph_.Initialize(graph_config_));
  MP_ASSERT_OK(graph_.StartRun({}));
  auto status = graph_.WaitUntilIdle();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(
      status.message(),
      ::testing::HasSubstr("only support applying FFT on mono channel"));
}

TEST_F(AudioToTensorCalculatorFftTest, TestImpulseSignal) {
  constexpr double sample_rate = 16000;
  ConfigGraph(1, 320, 160, sample_rate, 320);
  RunGraph(CreateImpulseSignalData(320, 160), sample_rate);
  for (int i = 0; i < tensors_packets_.size(); ++i) {
    const auto& tensors = tensors_packets_[i].Get<std::vector<Tensor>>();
    ASSERT_EQ(1, tensors.size());
    const Tensor& output_tensor =
        tensors_packets_[0].Get<std::vector<Tensor>>()[0];
    auto* buffer = output_tensor.GetCpuReadView().buffer<float>();
    int num_values = output_tensor.shape().num_elements();
    const std::vector<float> output_floats(buffer, buffer + num_values);
    // Impulse signal should have (approximately) const power across all
    // frequency bins.
    const auto& pair =
        dc_and_nyquist_packets_[i].Get<std::pair<float, float>>();
    EXPECT_FLOAT_EQ(pair.first, 1.0f);
    EXPECT_FLOAT_EQ(pair.second, 1.0f);
    for (int j = 0; j < num_values / 2; ++j) {
      std::complex<float> cf(output_floats[j * 2], output_floats[j * 2 + 1]);
      EXPECT_FLOAT_EQ(std::norm(cf), 1.0f);
    }
  }
  CloseGraph();
}

}  // namespace
}  // namespace mediapipe
