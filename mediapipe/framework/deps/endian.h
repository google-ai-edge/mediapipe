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

#ifndef MEDIAPIPE_FRAMEWORK_DEPS_ENDIAN_H_
#define MEDIAPIPE_FRAMEWORK_DEPS_ENDIAN_H_

#include <cstdint>
#include <cstring>

#include "absl/base/config.h"

namespace mediapipe {

inline bool IsLittleEndian() {
#ifdef ABSL_IS_LITTLE_ENDIAN
  return true;
#else
  return false;
#endif
}

namespace little_endian {

template <typename T>
inline T InternalLoadUnaligned(const void* p) {
  T t;
  std::memcpy(&t, p, sizeof(T));
  return t;
}

inline uint16_t Load16(const void* p) {
  return InternalLoadUnaligned<uint16_t>(p);
}

inline uint32_t Load32(const void* p) {
  return InternalLoadUnaligned<uint32_t>(p);
}

inline uint64_t Load64(const void* p) {
  return InternalLoadUnaligned<uint64_t>(p);
}

}  // namespace little_endian
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_DEPS_ENDIAN_H_
