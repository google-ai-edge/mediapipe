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

#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"

namespace mediapipe {
namespace api2 {

using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

// Aggregates EmbeddingResult packets into a vector of timestamped
// EmbeddingResult. Acts as a pass-through if no timestamp aggregation is
// needed.
//
// Inputs:
//   EMBEDDINGS: EmbeddingResult
//     The EmbeddingResult packets to aggregate.
//   TIMESTAMPS: std::vector<Timestamp> @Optional.
//     The collection of timestamps that this calculator should aggregate. This
//     stream is optional: if provided then the TIMESTAMPED_EMBEDDINGS output
//     will contain the aggregated results. Otherwise as no timestamp
//     aggregation is required the EMBEDDINGS output is used to pass the inputs
//     EmbeddingResults unchanged.
//
// Outputs:
//   EMBEDDINGS: EmbeddingResult @Optional
//     The input EmbeddingResult, unchanged. Must be connected if the TIMESTAMPS
//     input is not connected, as it signals that timestamp aggregation is not
//     required.
//  TIMESTAMPED_EMBEDDINGS: std::vector<EmbeddingResult> @Optional
//     The embedding results aggregated by timestamp. Must be connected if the
//     TIMESTAMPS input is connected as it signals that timestamp aggregation is
//     required.
//
// Example without timestamp aggregation (pass-through):
// node {
//   calculator: "EmbeddingAggregationCalculator"
//   input_stream: "EMBEDDINGS:embeddings_in"
//   output_stream: "EMBEDDINGS:embeddings_out"
// }
//
// Example with timestamp aggregation:
// node {
//   calculator: "EmbeddingAggregationCalculator"
//   input_stream: "EMBEDDINGS:embeddings_in"
//   input_stream: "TIMESTAMPS:timestamps_in"
//   output_stream: "TIMESTAMPED_EMBEDDINGS:timestamped_embeddings_out"
// }
class EmbeddingAggregationCalculator : public Node {
 public:
  static constexpr Input<EmbeddingResult> kEmbeddingsIn{"EMBEDDINGS"};
  static constexpr Input<std::vector<Timestamp>>::Optional kTimestampsIn{
      "TIMESTAMPS"};
  static constexpr Output<EmbeddingResult>::Optional kEmbeddingsOut{
      "EMBEDDINGS"};
  static constexpr Output<std::vector<EmbeddingResult>>::Optional
      kTimestampedEmbeddingsOut{"TIMESTAMPED_EMBEDDINGS"};
  MEDIAPIPE_NODE_CONTRACT(kEmbeddingsIn, kTimestampsIn, kEmbeddingsOut,
                          kTimestampedEmbeddingsOut);

  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);

 private:
  bool time_aggregation_enabled_;
  std::unordered_map<int64_t, EmbeddingResult> cached_embeddings_;
};

absl::Status EmbeddingAggregationCalculator::UpdateContract(
    CalculatorContract* cc) {
  if (kTimestampsIn(cc).IsConnected()) {
    RET_CHECK(kTimestampedEmbeddingsOut(cc).IsConnected());
  } else {
    RET_CHECK(kEmbeddingsOut(cc).IsConnected());
  }
  return absl::OkStatus();
}

absl::Status EmbeddingAggregationCalculator::Open(CalculatorContext* cc) {
  time_aggregation_enabled_ = kTimestampsIn(cc).IsConnected();
  return absl::OkStatus();
}

absl::Status EmbeddingAggregationCalculator::Process(CalculatorContext* cc) {
  if (time_aggregation_enabled_) {
    cached_embeddings_[cc->InputTimestamp().Value()] =
        std::move(*kEmbeddingsIn(cc));
    if (kTimestampsIn(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    auto timestamps = kTimestampsIn(cc).Get();
    std::vector<EmbeddingResult> results;
    results.reserve(timestamps.size());
    for (const auto& timestamp : timestamps) {
      auto& result = cached_embeddings_[timestamp.Value()];
      result.set_timestamp_ms((timestamp.Value() - timestamps[0].Value()) /
                              1000);
      results.push_back(std::move(result));
      cached_embeddings_.erase(timestamp.Value());
    }
    kTimestampedEmbeddingsOut(cc).Send(std::move(results));
  } else {
    auto result = kEmbeddingsIn(cc).Get();
    result.set_timestamp_ms(cc->InputTimestamp().Value() / 1000);
    kEmbeddingsOut(cc).Send(result);
  }
  RET_CHECK(cached_embeddings_.empty());
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(EmbeddingAggregationCalculator);

}  // namespace api2
}  // namespace mediapipe
