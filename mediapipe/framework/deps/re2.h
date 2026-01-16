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

#ifndef MEDIAPIPE_FRAMEWORK_DEPS_RE2_H_
#define MEDIAPIPE_FRAMEWORK_DEPS_RE2_H_

#include <regex>  // NOLINT

#include "absl/base/call_once.h"

namespace mediapipe {

// Implements a subset of RE2 using std::regex_match.
class RE2 {
 public:
  RE2(const std::string& pattern) : std_regex_(pattern) {}
  static bool FullMatch(std::string text, const RE2& re) {
    return std::regex_match(text, re.std_regex_);
  }
  static bool PartialMatch(std::string text, const RE2& re) {
    return std::regex_search(text, re.std_regex_);
  }
  static int GlobalReplace(std::string* text, const RE2& re,
                           const std::string& rewrite) {
    std::smatch sm;
    std::regex_search(*text, sm, re.std_regex_);
    *text = std::regex_replace(*text, re.std_regex_, rewrite);
    return std::max(0, static_cast<int>(sm.size()) - 1);
  }

 private:
  std::regex std_regex_;
};

// Implements LazyRE2 using absl::call_once.
class LazyRE2 {
 public:
  RE2& operator*() const {
    absl::call_once(once_, [&]() { ptr_ = new RE2(pattern_); });
    return *ptr_;
  }
  RE2* operator->() const { return &**this; }
  const char* pattern_;
  mutable RE2* ptr_;
  mutable absl::once_flag once_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_DEPS_RE2_H_
