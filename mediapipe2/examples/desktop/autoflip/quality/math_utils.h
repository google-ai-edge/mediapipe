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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_MATH_UTILS_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_MATH_UTILS_H_

class MathUtil {
 public:
  // Clamps value to the range [low, high].  Requires low <= high. Returns false
  // if this check fails, otherwise returns true. Caller should first check the
  // returned boolean.
  template <typename T>  // T models LessThanComparable.
  static bool Clamp(const T& low, const T& high, const T& value, T* result) {
    // Prevents errors in ordering the arguments.
    if (low > high) {
      return false;
    }
    if (high < value) {
      *result = high;
    } else if (value < low) {
      *result = low;
    } else {
      *result = value;
    }
    return true;
  }
};

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_MATH_UTILS_H_
