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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_HAS_UPDATE_CONTRACT_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_HAS_UPDATE_CONTRACT_H_

#include <type_traits>

namespace mediapipe::api3 {

template <typename I, typename C, typename = std::void_t<>>
struct HasUpdateContractHelper : std::false_type {};

template <typename I, typename C>
struct HasUpdateContractHelper<
    I, C, std::void_t<decltype(I::UpdateContract(std::declval<C&>()))>>
    : std::true_type {};

template <typename I, typename C>
static constexpr bool kHasUpdateContract = HasUpdateContractHelper<I, C>{};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_HAS_UPDATE_CONTRACT_H_
