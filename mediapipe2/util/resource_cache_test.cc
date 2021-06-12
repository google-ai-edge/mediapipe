// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/util/resource_cache.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

#define EXPECT_BETWEEN(low, high, value) \
  do {                                   \
    EXPECT_LE((low), (value));           \
    EXPECT_GE((high), (value));          \
  } while (0)

namespace mediapipe {
namespace {

using ::testing::_;
using ::testing::MockFunction;
using ::testing::Return;

using IntCache = ResourceCache<int, std::shared_ptr<int>, std::hash<int>>;
using MockCreate =
    MockFunction<std::shared_ptr<int>(const int& key, int request_count)>;

TEST(ResourceCacheTest, ReturnsNull) {
  IntCache cache;
  MockCreate create;

  EXPECT_CALL(create, Call(1, 1)).WillOnce(Return(nullptr));
  EXPECT_EQ(nullptr, cache.Lookup(1, create.AsStdFunction()));
}

TEST(ResourceCacheTest, CountsRequests) {
  IntCache cache;
  MockCreate create11;
  MockCreate create12;
  MockCreate create21;

  EXPECT_CALL(create11, Call(1, 1)).WillOnce(Return(nullptr));
  EXPECT_CALL(create12, Call(1, 2)).WillOnce(Return(nullptr));
  EXPECT_CALL(create11, Call(2, 1)).WillOnce(Return(nullptr));

  // Verify that request counts are updated, and separate by key.
  EXPECT_EQ(nullptr, cache.Lookup(1, create11.AsStdFunction()));
  EXPECT_EQ(nullptr, cache.Lookup(1, create12.AsStdFunction()));
  EXPECT_EQ(nullptr, cache.Lookup(2, create11.AsStdFunction()));
}

TEST(ResourceCacheTest, CachesValues) {
  IntCache cache;
  auto value1 = std::make_shared<int>(1);
  auto value2 = std::make_shared<int>(2);

  MockCreate create1;
  MockCreate create2;
  MockCreate no_create;

  EXPECT_CALL(create1, Call(1, 1)).WillOnce(Return(value1));
  EXPECT_CALL(create2, Call(2, 1)).WillOnce(Return(value2));
  EXPECT_CALL(no_create, Call(_, _)).Times(0);
  // Calls creating the values.
  EXPECT_EQ(value1, cache.Lookup(1, create1.AsStdFunction()));
  EXPECT_EQ(value2, cache.Lookup(2, create2.AsStdFunction()));

  // Calls returning existing values.
  EXPECT_EQ(value1, cache.Lookup(1, no_create.AsStdFunction()));
  EXPECT_EQ(value2, cache.Lookup(2, no_create.AsStdFunction()));
}

TEST(ResourceCacheTest, EvictToMaxSize) {
  IntCache cache;
  MockCreate create;

  EXPECT_CALL(create, Call(_, 1))
      .WillRepeatedly([](int key, int request_count) {
        return std::make_shared<int>(key);
      });

  // Add three entries.
  EXPECT_NE(nullptr, cache.Lookup(1, create.AsStdFunction()));
  EXPECT_NE(nullptr, cache.Lookup(2, create.AsStdFunction()));
  EXPECT_NE(nullptr, cache.Lookup(3, create.AsStdFunction()));

  // Keep only two.
  auto evicted = cache.Evict(/*max_count=*/2,
                             /*request_count_scrub_interval=*/4);
  ASSERT_EQ(1, evicted.size());
  int evicted_entry = *evicted[0];
  EXPECT_BETWEEN(1, 3, evicted_entry);

  MockCreate no_create;
  EXPECT_CALL(no_create, Call(_, 1)).WillOnce(Return(nullptr));
  EXPECT_EQ(nullptr, cache.Lookup(evicted_entry, no_create.AsStdFunction()));
  for (int key = 1; key <= 3; key++) {
    if (key != evicted_entry) {
      EXPECT_NE(nullptr, cache.Lookup(key, no_create.AsStdFunction()));
    }
  }
}

TEST(ResourceCacheTest, EvictWithScrub) {
  IntCache cache;
  MockCreate create;

  EXPECT_CALL(create, Call(_, 1))
      .WillRepeatedly([](int key, int request_count) {
        return std::make_shared<int>(key);
      });

  EXPECT_NE(nullptr, cache.Lookup(1, create.AsStdFunction()));
  EXPECT_NE(nullptr, cache.Lookup(2, create.AsStdFunction()));
  EXPECT_NE(nullptr, cache.Lookup(3, create.AsStdFunction()));

  // 3 entries, total request count 4, so nothing evicted from this call.
  EXPECT_TRUE(
      cache.Evict(/*max_count=*/3, /*request_count_scrub_interval=*/4).empty());

  // Increment request counts.
  EXPECT_NE(nullptr, cache.Lookup(1, create.AsStdFunction()));
  EXPECT_NE(nullptr, cache.Lookup(3, create.AsStdFunction()));

  // Expected to evict entry 2, and halve request counts for the other two
  // entries.
  auto evicted =
      cache.Evict(/*max_count=*/3, /*request_count_scrub_interval=*/5);
  ASSERT_EQ(1, evicted.size());
  EXPECT_EQ(2, *evicted[0]);

  // Increment request count.
  EXPECT_NE(nullptr, cache.Lookup(3, create.AsStdFunction()));
  // Expected to evict entry 1.
  evicted = cache.Evict(/*max_count=*/3, /*request_count_scrub_interval=*/1);
  ASSERT_EQ(1, evicted.size());
  EXPECT_EQ(1, *evicted[0]);
}

}  // namespace
}  // namespace mediapipe
