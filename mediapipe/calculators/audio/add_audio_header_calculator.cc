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

#include "mediapipe/calculators/audio/add_audio_header_calculator.h"

#include <memory>

#include "absl/status/status.h"
#include "mediapipe/calculators/audio/add_audio_header_calculator_options.pb.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"

namespace mediapipe {

namespace {
static constexpr double kDefaultSampleRate = 48000.0;
static constexpr int kDefaultNumChannels = 1;
}  // namespace

class AddAudioHeaderCalculatorImpl
    : public api3::Calculator<AddAudioHeaderNode,
                              AddAudioHeaderCalculatorImpl> {
 public:
  absl::Status Open(api3::CalculatorContext<AddAudioHeaderNode>& cc) final {
    const auto& options = cc.options.Get();
    auto output_header = std::make_unique<mediapipe::TimeSeriesHeader>();
    output_header->set_num_channels(options.num_channels() == 0
                                        ? kDefaultNumChannels
                                        : options.num_channels());
    output_header->set_sample_rate(options.sample_rate() == 0
                                       ? kDefaultSampleRate
                                       : options.sample_rate());

    // TODO: b/503083496 - update when API3 support headers natively.
    cc.GetGenericContext().Outputs().Index(0).SetHeader(
        mediapipe::Adopt(output_header.release()));

    return absl::OkStatus();
  }

  absl::Status Process(api3::CalculatorContext<AddAudioHeaderNode>& cc) final {
    if (cc.in.IsEmpty()) {
      return absl::OkStatus();
    }
    cc.out.Send(cc.in.Packet());
    return absl::OkStatus();
  }
};

}  // namespace mediapipe
