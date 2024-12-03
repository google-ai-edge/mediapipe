/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Forked from a test library written by Jyrki Alakuijala.
// Original copyright message below.
// Copyright 2009 Google Inc.
// Author: jyrki@google.com (Jyrki Alakuijala)
//
// Tests for the fast hashing algorithm based on Austin Appleby's
// MurmurHash 2.0 algorithm. See http://murmurhash.googlepages.com/

#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/hash/murmur.h"

#include <string.h>

#include <cstdint>
#include <string>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::text::language_detector::custom_ops::hash {

TEST(Murmur, EmptyData64) {
  EXPECT_EQ(uint64_t{0}, MurmurHash64WithSeed(nullptr, uint64_t{0}, 0));
}

TEST(Murmur, VaryWithDifferentSeeds) {
  // While in theory different seeds could return the same
  // hash for the same data this is unlikely.
  char data1 = 'x';
  EXPECT_NE(MurmurHash64WithSeed(&data1, 1, 100),
            MurmurHash64WithSeed(&data1, 1, 101));
}

// Hashes don't change.
TEST(Murmur, Idempotence) {
  const char data[] = "deadbeef";
  const size_t dlen = strlen(data);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(MurmurHash64WithSeed(data, dlen, i),
              MurmurHash64WithSeed(data, dlen, i));
  }

  const char next_data[] = "deadbeef000---";
  const size_t next_dlen = strlen(next_data);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(MurmurHash64WithSeed(next_data, next_dlen, i),
              MurmurHash64WithSeed(next_data, next_dlen, i));
  }
}
}  // namespace mediapipe::tasks::text::language_detector::custom_ops::hash
