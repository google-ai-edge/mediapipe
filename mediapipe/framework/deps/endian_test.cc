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

#include "mediapipe/framework/deps/endian.h"

#include <cstdint>

#include "absl/base/config.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

const uint16_t k16Value{0x0123};
const uint32_t k32Value{0x01234567};
const uint64_t k64Value{0x0123456789abcdef};

TEST(EndianTest, IsLittleEndianWorks) {
#ifdef ABSL_IS_LITTLE_ENDIAN
  EXPECT_TRUE(IsLittleEndian());
#else
  EXPECT_FALSE(IsLittleEndian());
#endif
}

TEST(EndianTest, LoadWorks) {
  const uint16_t u16Buf = k16Value;
  EXPECT_EQ(little_endian::Load16(&u16Buf), k16Value);

  const uint32_t u32Buf = k32Value;
  EXPECT_EQ(little_endian::Load32(&u32Buf), k32Value);

  const uint64_t u64Buf = k64Value;
  EXPECT_EQ(little_endian::Load64(&u64Buf), k64Value);
}

}  // namespace
}  // namespace mediapipe
