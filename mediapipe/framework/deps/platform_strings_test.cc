// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/framework/deps/platform_strings.h"

#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

TEST(PlatformStrings, ThereAndBack) {
  const std::string source = "Шчучыншчына";
  const std::string result = NativeToUtf8(Utf8ToNative(source));
  EXPECT_EQ(result, source);
}

}  // namespace
}  // namespace mediapipe
