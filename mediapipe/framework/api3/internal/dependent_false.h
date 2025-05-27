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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_DEPENDENT_FALSE_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_DEPENDENT_FALSE_H_

#include <type_traits>

namespace mediapipe::api3 {

// Workaround for static_assert(false). Example:
//   dependent_false<T>::value returns false.
// For more information, see:
// https://en.cppreference.com/w/cpp/language/if#Constexpr_If
// TODO: migrate to a common utility when available.
template <class T>
struct dependent_false : std::false_type {};

template <class T>
inline constexpr auto dependent_false_v = dependent_false<T>{};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_DEPENDENT_FALSE_H_
