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

// Tests for Packet that need to be compiled with -Werror.

// We want Packet to call the correct form of delete ("delete[]" for pointers
// to arrays, "delete" for pointers to scalars) depending on the type of the
// contents. Calling the wrong form is incorrect. Clang recognizes the mistake
// and outputs a warning (warning: 'delete' applied to a pointer-to-array type
// '...' treated as delete[]). To avoid ignoring it, we must compile this file
// with -Werror.
// As of today, there is no flag controlling that specific warning (see
// https://github.com/llvm-mirror/clang/blob/c3f7ae01e2cd11964e4aa3b49622ae9419023fa7/test/Misc/warning-flags.c#L62),
// therefore we cannot use the -Werror=... form, and have to turn -Werror on
// for all warnings instead. To avoid interference from possible warnings from
// other tests, these tests are put in a separate file.

#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

class LiveObjectsCounter {
 public:
  LiveObjectsCounter() { ++counter_; }
  LiveObjectsCounter(const LiveObjectsCounter&) = delete;
  LiveObjectsCounter& operator=(const LiveObjectsCounter&) = delete;
  ~LiveObjectsCounter() { --counter_; }
  static int counter() { return counter_; }

 private:
  static int counter_;
};

int LiveObjectsCounter::counter_ = 0;

TEST(Packet, DeletesNonArray) {
  ASSERT_EQ(0, LiveObjectsCounter::counter());
  {
    Packet packet = Adopt(new LiveObjectsCounter);
    EXPECT_EQ(1, LiveObjectsCounter::counter());
  }
  EXPECT_EQ(0, LiveObjectsCounter::counter());
}

TEST(Packet, DeletesBoundedArray) {
  ASSERT_EQ(0, LiveObjectsCounter::counter());
  {
    // new T[3] returns a T*, so we must cast it to get the right type.
    Packet packet = Adopt(
        reinterpret_cast<LiveObjectsCounter(*)[3]>(new LiveObjectsCounter[3]));
    EXPECT_EQ(3, LiveObjectsCounter::counter());
  }
  EXPECT_EQ(0, LiveObjectsCounter::counter());
}

TEST(Packet, DeletesUnboundedArray) {
  for (int size = 0; size < 10; ++size) {
    ASSERT_EQ(0, LiveObjectsCounter::counter());
    {
      // new T[size] returns a T*, so we must cast it to get the right type.
      Packet packet = Adopt(reinterpret_cast<LiveObjectsCounter(*)[]>(
          new LiveObjectsCounter[size]));
      EXPECT_EQ(size, LiveObjectsCounter::counter());
    }
    EXPECT_EQ(0, LiveObjectsCounter::counter());
  }
}

}  // namespace
}  // namespace mediapipe
