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

#ifndef MEDIAPIPE_CALCULATORS_AUDIO_ADD_AUDIO_HEADER_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_AUDIO_ADD_AUDIO_HEADER_CALCULATOR_H_

#include "mediapipe/calculators/audio/add_audio_header_calculator_options.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/matrix.h"

namespace mediapipe {

// A calculator that adds a TimeSeriesHeader to an audio stream.
// The sample rate and number of channels are provided via
// AddAudioHeaderCalculatorOptions. If not provided, sample_rate defaults to
// 48000.0 and num_channels to 1.
//
// Example:
// node {
//   calculator: "AddAudioHeaderCalculator"
//   input_stream: "audio_in"
//   output_stream: "audio_out"
//   node_options: {
//     [type.googleapis.com/mediapipe.AddAudioHeaderCalculatorOptions] {
//       sample_rate: 16000.0
//       num_channels: 2
//     }
//   }
// }
struct AddAudioHeaderNode : mediapipe::api3::Node<"AddAudioHeaderCalculator"> {
  template <typename S>
  struct Contract {
    // Input audio data.
    api3::Input<S, Matrix> in{""};
    // Audio data with TimeSeriesHeader attached.
    api3::Output<S, Matrix> out{""};
    // Options for the TimeSeriesHeader, such as sample rate and number of
    // channels.
    api3::Options<S, AddAudioHeaderCalculatorOptions> options{};
  };
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_AUDIO_ADD_AUDIO_HEADER_CALCULATOR_H_
