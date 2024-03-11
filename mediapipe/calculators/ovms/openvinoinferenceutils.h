//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <string>
#include <vector>
namespace mediapipe {

// Function from ovms/src/string_utils.h
bool startsWith(const std::string& str, const std::string& prefix);

// Function from ovms/src/string_utils.h
std::vector<std::string> tokenize(const std::string& str, const char delimiter);

// Function from ovms/src/string_utils.h
bool endsWith(const std::string& str, const std::string& match);

}  // namespace mediapipe
