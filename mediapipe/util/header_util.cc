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

#include "mediapipe/util/header_util.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

::mediapipe::Status CopyInputHeadersToOutputs(const InputStreamSet& inputs,
                                              const OutputStreamSet& outputs) {
  for (auto id = inputs.BeginId(); id < inputs.EndId(); ++id) {
    std::pair<std::string, int> tag_index = inputs.TagAndIndexFromId(id);
    auto output_id = outputs.GetId(tag_index.first, tag_index.second);
    if (output_id.IsValid()) {
      outputs.Get(output_id)->SetHeader(inputs.Get(id)->Header());
    }
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CopyInputHeadersToOutputs(const InputStreamShardSet& inputs,
                                              OutputStreamShardSet* outputs) {
  for (auto id = inputs.BeginId(); id < inputs.EndId(); ++id) {
    std::pair<std::string, int> tag_index = inputs.TagAndIndexFromId(id);
    auto output_id = outputs->GetId(tag_index.first, tag_index.second);
    if (output_id.IsValid()) {
      outputs->Get(output_id).SetHeader(inputs.Get(id).Header());
    }
  }

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
