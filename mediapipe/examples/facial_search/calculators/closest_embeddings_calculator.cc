// Copyright 2020 The MediaPipe Authors.
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

#include <cmath>

#include "mediapipe/examples/facial_search/calculators/closest_embeddings_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#define COUNTOF(ARRAY) (sizeof(ARRAY) / sizeof(ARRAY[0]))

namespace mediapipe {

namespace {
constexpr char kClassifications[] = "CLASSIFICATIONS";
constexpr char kCollection[] = "COLLECTION";
constexpr char kLabels[] = "LABELS";
constexpr char kFloats[] = "FLOATS";
typedef std::vector<Classification> Classifications;
typedef std::vector<float> Floats;
typedef std::vector<Floats> Collection;
typedef std::vector<std::string> Labels;
}  // namespace

// Given a flat vector of embeddings, finds the top k closests vectors
// from the embeddings collection and returns the value associated with these
// vectors.
//
// Inputs:
//   FLOATS: the input embedding to compare, as an std::vector of floats.
//   COLLECTION: an std::vector of embeddings that this calculcator compares
//   FLOATS against. LABELS: an (optional) std::vector of strings whose indices
//   match COLLECTION's. CLASSIFICATIONS: the k-closest embeddings as an
//   std::vector of Classification.s where
//                    Classification.id is the embedding's index in COLLECTION,
//                    Classification.label is the embedding's index in LABELS
//                    and Classification.score is the distance between the two
//                    embeddings.
//
// Options:
//   top_k: number of embeddings closest to input to search for.
//
// Notes:
//   * The distance function used by default is the Euclidian distance.
//   * Every vector in COLLECTION must have the same dimension as the input
//   vector.
//   * When given an empty input vector, an empty output vector is returned.
//
// Usage example:
// node {
//   calculator: "ClosestEmbeddingsCalculator"
//   input_side_packet: "COLLECTION:embeddings_collection"
//   input_side_packet: "LABELS:collection_labels"
//   input_stream: "FLOATS:embeddings_vector"
//   output_stream: "CLASSIFICATIONS:memes"
//   options: {
//     [mediapipe.ClosestEmbeddingsCalculatorOptions.ext]: {
//       top_k: 3
//     }
//   }
// }
class ClosestEmbeddingsCalculator : public CalculatorBase {
 public:
  ClosestEmbeddingsCalculator() {}
  ~ClosestEmbeddingsCalculator() override {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->InputSidePackets().HasTag(kCollection));
    cc->InputSidePackets().Tag(kCollection).Set<Collection>();
    if (cc->InputSidePackets().HasTag(kLabels))
      cc->InputSidePackets().Tag(kLabels).Set<Labels>();

    RET_CHECK(cc->Inputs().HasTag(kFloats));
    cc->Inputs().Tag(kFloats).Set<Floats>();

    RET_CHECK(cc->Outputs().HasTag(kClassifications));
    cc->Outputs().Tag(kClassifications).Set<Classifications>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    collection_ = cc->InputSidePackets().Tag(kCollection).Get<Collection>();
    RET_CHECK(!collection_.empty());
    if (cc->InputSidePackets().HasTag(kLabels)) {
      labels_ = cc->InputSidePackets().Tag(kLabels).Get<Labels>();
      RET_CHECK_EQ(labels_.size(), collection_.size());
    }
    const auto& options = cc->Options<ClosestEmbeddingsCalculatorOptions>();
    top_k_ = std::min(size_t(options.top_k()), collection_.size());
    RET_CHECK_NE(top_k_, 0);
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    const auto input = cc->Inputs().Tag(kFloats).Get<Floats>();
    auto output = absl::make_unique<Classifications>();
    if (input.empty()) {
      cc->Outputs()
          .Tag(kClassifications)
          .Add(output.release(), cc->InputTimestamp());
      return ::mediapipe::OkStatus();
    }

    RET_CHECK_EQ(input.size(), collection_[0].size())
        << "Embeddings should have " << collection_[0].size()
        << " dimensions. Got " << input.size() << " floats.";

    output->reserve(top_k_);
    RET_CHECK_OK(AppendKClosest(input, output.get()));

    cc->Outputs()
        .Tag(kClassifications)
        .Add(output.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }

 private:
  size_t top_k_;
  Collection collection_;
  Labels labels_;

  ::mediapipe::Status AppendKClosest(const Floats& input,
                                     Classifications* best) {
    auto matches = Classifications();
    matches.reserve(top_k_);
    for (int i = 0; i < collection_.size(); ++i) {
      const auto& item = collection_[i];
      RET_CHECK_EQ(item.size(), input.size());
      Classification classification;
      classification.set_index(i);
      classification.set_score(EuclidianDistance(item, input));
      if (!labels_.empty()) classification.set_label(labels_[i]);
      matches.emplace_back(classification);
    }
    TopK(matches);
    for (const auto match : matches) {
      best->emplace_back(match);
    }
    return ::mediapipe::OkStatus();
  }

  float EuclidianDistance(const Floats& a, const Floats& b) {
    float sum = 0;
    for (int i = 0; i < a.size(); ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
  }

  void TopK(Classifications& classifications) {
    if (classifications.size() > 1 && classifications.size() > top_k_) {
      std::partial_sort(classifications.begin(),
                        classifications.begin() + top_k_, classifications.end(),
                        [](const Classification& a, const Classification& b) {
                          // Sorts such that results closest to zero are at the
                          // top
                          return a.score() < b.score();
                        });
      classifications.erase(classifications.begin() + top_k_,
                            classifications.end());
    }
  }
};
REGISTER_CALCULATOR(ClosestEmbeddingsCalculator);

}  // namespace mediapipe
