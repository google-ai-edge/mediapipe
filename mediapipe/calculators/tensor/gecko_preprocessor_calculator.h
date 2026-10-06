// Copyright 2026 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_GECKO_PREPROCESSOR_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_GECKO_PREPROCESSOR_CALCULATOR_H_

#include <string>
#include <vector>

#include "mediapipe/calculators/tensor/gecko_preprocessor_calculator.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

namespace mediapipe {

// Preprocesses input text into one int32 input tensor for a Gecko model using
// a SentencePieceTokenizer.
//
// Example:
// node {
//   calculator: "GeckoPreprocessorCalculator"
//   input_stream: "TEXT:text"
//   input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
//   output_stream: "TENSORS:tensors"
//   node_options: {
//     [type.googleapis.com/mediapipe.GeckoPreprocessorCalculatorOptions] {
//       max_seq_len: 256
//     }
//   }
// }
struct GeckoPreprocessorCalculatorNode
    : public api3::Node<"GeckoPreprocessorCalculator"> {
  template <typename S>
  struct Contract {
    // The input text to preprocess.
    api3::Input<S, std::string> text_in{"TEXT"};
    // The metadata extractor for the text model, used to extract the metadata
    // to construct the SentencePieceTokenizer.
    api3::SideInput<S, tasks::metadata::ModelMetadataExtractor>
        metadata_extractor{"METADATA_EXTRACTOR"};
    // Vector containing the preprocessed input tensors for the text model.
    api3::Output<S, std::vector<Tensor>> tensors_out{"TENSORS"};
    // Options for the GeckoPreprocessorCalculator.
    api3::Options<S, GeckoPreprocessorCalculatorOptions> options{};
  };
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_GECKO_PREPROCESSOR_CALCULATOR_H_
