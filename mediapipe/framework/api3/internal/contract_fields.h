// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_FIELDS_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_FIELDS_H_

namespace mediapipe::api3 {

// Used internally by the framework to distinguish templated (side) inputs,
// (side) outputs and options.

struct InputStreamField {};
struct OutputStreamField {};
struct InputSidePacketField {};
struct OutputSidePacketField {};
struct RepeatedField {};
struct OptionalField {};
struct OptionsField {};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_FIELDS_H_
