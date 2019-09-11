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

#include "mediapipe/framework/tool/validate.h"

#include <string>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/validate_name.h"

namespace mediapipe {

namespace tool {

::mediapipe::Status ValidateInput(const InputCollection& input_collection) {
  if (!input_collection.name().empty()) {
    MP_RETURN_IF_ERROR(tool::ValidateName(input_collection.name())).SetPrepend()
        << "InputCollection " << input_collection.name()
        << " has improperly specified name: ";
  }
  if (input_collection.input_type() <= InputCollection::UNKNOWN ||
      input_collection.input_type() >= InputCollection::INVALID_UPPER_BOUND) {
    return ::mediapipe::InvalidArgumentError(
        "InputCollection must specify a valid input_type.");
  }
  if (input_collection.file_name().empty()) {
    return ::mediapipe::InvalidArgumentError(
        "InputCollection must specify a file_name.");
  }
  return ::mediapipe::OkStatus();
}

}  // namespace tool
}  // namespace mediapipe
