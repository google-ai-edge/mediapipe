// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/framework/profiler/circular_buffer.h"

#include "absl/time/time.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/threadpool.h"

namespace {

class CircularBufferTest : public ::testing::Test {
 protected:
  // Called before every TEST_F using this fixture.
  CircularBufferTest() {}
};

TEST_F(CircularBufferTest, SequentialWriteAndRead) {
  mediapipe::CircularBuffer<std::string> my_buffer(100);
  my_buffer.push_back("one");
  my_buffer.push_back("two");
  my_buffer.push_back("three");
  std::vector<std::string> snapshot;
  auto begin = my_buffer.begin();
  auto end = my_buffer.end();
  for (auto iter = begin; iter < end; ++iter) {
    snapshot.push_back(*iter);
  }
  std::vector<std::string> expected = {"one", "two", "three"};
  EXPECT_EQ(snapshot, expected);
  begin = end;
  end = my_buffer.end();
  for (auto iter = begin; iter < end; ++iter) {
    snapshot.push_back(*iter);
  }
  EXPECT_EQ(snapshot, expected);
}

TEST_F(CircularBufferTest, ParallelWriteAndRead) {
  mediapipe::CircularBuffer<std::string> buffer(100);
  auto first = buffer.begin();
  std::atomic_int read_sum(0);
  std::atomic_int read_count(0);
  {
    mediapipe::ThreadPool pool(12);
    pool.StartWorkers();

    // Start 6 writers.
    for (int w = 0; w < 6; ++w) {
      pool.Schedule([&]() {
        for (int i = 0; i < 300; ++i) {
          buffer.push_back("w5");
          absl::SleepFor(absl::Microseconds(1));
        }
      });
    }

    // Start 6 readers.
    for (int w = 0; w < 6; ++w) {
      pool.Schedule([&]() {
        for (int t = 0; t < 10; ++t) {
          while (buffer.end() - buffer.begin() < 50) {
          }
          auto end = buffer.end();
          for (auto it = buffer.begin(); it < end; ++it) {
            read_sum += (*it).size();
            read_count++;
          }
        }
      });
    }
  }

  // Validate the total number of writes including failed writes.
  EXPECT_EQ(1800, buffer.end() - first);
  // Validate that every read succeeds.
  EXPECT_LT(2000, read_count);
  EXPECT_EQ(read_sum, read_count * 2);
}

TEST_F(CircularBufferTest, SequentialGetWraps) {
  mediapipe::CircularBuffer<int> buffer(3);
  buffer.push_back(2);
  ASSERT_EQ(2, buffer.Get(0));
  ASSERT_EQ(*buffer.begin(), buffer.Get(0));
  buffer.push_back(3);
  ASSERT_EQ(2, buffer.Get(0));
  ASSERT_EQ(3, buffer.Get(1));
  ASSERT_EQ(*buffer.begin(), buffer.Get(0));
  for (int i = 2; i < 100; ++i) {
    buffer.push_back(i + 2);
    ASSERT_EQ(i + 2, buffer.Get(2));
    ASSERT_EQ(i, buffer.Get(0));
    ASSERT_EQ(*buffer.begin(), buffer.Get(0));
  }
}

}  // namespace
