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

#include "mediapipe/util/resource_util.h"

#include <iostream>

#include "absl/strings/str_split.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util_custom.h"
#include "mediapipe/util/resource_util_internal.h"

namespace mediapipe {

namespace {
ResourceProviderFn resource_provider_ = nullptr;
}  // namespace

absl::Status GetResourceContents(const std::string& path, std::string* output,
                                 bool read_as_binary) {
  if (resource_provider_) {
    return resource_provider_(path, output);
  }
  return internal::DefaultGetResourceContents(path, output, read_as_binary);
}

void SetCustomGlobalResourceProvider(ResourceProviderFn fn) {
  resource_provider_ = std::move(fn);
}

}  // namespace mediapipe
