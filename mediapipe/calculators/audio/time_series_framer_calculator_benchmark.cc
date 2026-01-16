// Copyright 2023 The MediaPipe Authors.
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
//
// Benchmark for TimeSeriesFramerCalculator.
#include <memory>
#include <random>
#include <vector>

#include "absl/log/absl_check.h"
#include "benchmark/benchmark.h"
#include "mediapipe/calculators/audio/time_series_framer_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/packet.h"

using ::mediapipe::Matrix;

void BM_TimeSeriesFramerCalculator(benchmark::State& state) {
  constexpr float kSampleRate = 32000.0;
  constexpr int kNumChannels = 2;
  constexpr int kFrameDurationSeconds = 5.0;
  std::mt19937 rng(0 /*seed*/);
  // Input around a half second's worth of samples at a time.
  std::uniform_int_distribution<int> input_size_dist(15000, 17000);
  // Generate a pool of random blocks of samples up front.
  std::vector<Matrix> sample_pool;
  sample_pool.reserve(20);
  for (int i = 0; i < 20; ++i) {
    sample_pool.push_back(Matrix::Random(kNumChannels, input_size_dist(rng)));
  }
  std::uniform_int_distribution<int> pool_index_dist(0, sample_pool.size() - 1);

  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input");
  config.add_output_stream("output");
  auto* node = config.add_node();
  node->set_calculator("TimeSeriesFramerCalculator");
  node->add_input_stream("input");
  node->add_output_stream("output");
  mediapipe::TimeSeriesFramerCalculatorOptions* options =
      node->mutable_options()->MutableExtension(
          mediapipe::TimeSeriesFramerCalculatorOptions::ext);
  options->set_frame_duration_seconds(kFrameDurationSeconds);

  for (auto _ : state) {
    state.PauseTiming();  // Pause benchmark timing.

    // Prepare input packets of random blocks of samples.
    std::vector<mediapipe::Packet> input_packets;
    input_packets.reserve(32);
    float t = 0;
    for (int i = 0; i < 32; ++i) {
      auto samples =
          std::make_unique<Matrix>(sample_pool[pool_index_dist(rng)]);
      const int num_samples = samples->cols();
      input_packets.push_back(mediapipe::Adopt(samples.release())
                                  .At(mediapipe::Timestamp::FromSeconds(t)));
      t += num_samples / kSampleRate;
    }
    // Initialize graph.
    mediapipe::CalculatorGraph graph;
    ABSL_CHECK_OK(graph.Initialize(config));
    // Prepare input header.
    auto header = std::make_unique<mediapipe::TimeSeriesHeader>();
    header->set_sample_rate(kSampleRate);
    header->set_num_channels(kNumChannels);

    state.ResumeTiming();  // Resume benchmark timing.

    ABSL_CHECK_OK(graph.StartRun({}, {{"input", Adopt(header.release())}}));
    for (auto& packet : input_packets) {
      ABSL_CHECK_OK(graph.AddPacketToInputStream("input", packet));
    }
    ABSL_CHECK(!graph.HasError());
    ABSL_CHECK_OK(graph.CloseAllInputStreams());
    ABSL_CHECK_OK(graph.WaitUntilIdle());
  }
}
BENCHMARK(BM_TimeSeriesFramerCalculator);

BENCHMARK_MAIN();
