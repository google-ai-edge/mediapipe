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

#include "mediapipe/framework/profiler/sharded_map.h"

#include <functional>

#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/threadpool.h"

namespace {

// Tests writing, reading and erasing in an unordered map.
template <class Map>
void TestWriteAndRead(Map& time_map) {
  time_map.insert({1110111, 22222222});
  int value = time_map.find(1110111)->second;
  time_map.insert({1113111, value});
  auto iter = time_map.find(1110111);
  time_map.erase(iter);
  iter = time_map.end();
  EXPECT_EQ(time_map.end(), time_map.find(1110111));
  EXPECT_NE(time_map.end(), time_map.find(1113111));
  EXPECT_EQ(22222222, time_map.find(1113111)->second);
  EXPECT_EQ(0, time_map.count(1110111));
  EXPECT_EQ(1, time_map.count(1113111));
  EXPECT_EQ(1, time_map.size());

  for (auto it = time_map.begin(); it != time_map.end(); ++it) {
    EXPECT_EQ(1113111, it->first);
    EXPECT_EQ(22222222, it->second);
  }

  iter = time_map.find(1113111);
  time_map.erase(iter);
  EXPECT_EQ(0, time_map.size());
  iter = time_map.end();
}

// Tests writing, reading and erasing in a ShardedMap.
TEST(ShardedMapTest, TestWriteAndRead) {
  absl::node_hash_map<int64, int64> simple_map;
  TestWriteAndRead(simple_map);
  ShardedMap<int64, int64> safe_map(4999, 1);
  TestWriteAndRead(safe_map);
  ShardedMap<int64, int64> sharded_map(4999);
  TestWriteAndRead(sharded_map);
}

// Starts 12 worker threads.
// Each worker thread repeats 1000 times:
// - writes 1 random key
// - reads 10 random keys
// - erases 1 random key
// Returns when all worker threads are done.
template <class Map>
void TestParallelAccess(Map& time_map, int num_threads) {
  int64 kNumTasks = 12;
  int64 kMaxKey = 9901;
  int64 kKeyStep = 1234;
  int64 kNumWrites = 1000;
  int64 kNumReads = 10;

  mediapipe::ThreadPool pool(num_threads);
  pool.StartWorkers();
  for (int i = 0; i < kNumTasks; ++i) {
    pool.Schedule([=, &time_map]() {
      int64 next_key = i * kNumWrites * kNumReads * kKeyStep % kMaxKey;
      for (int j = 0; j < kNumWrites; ++j) {
        // One map write.
        time_map.insert({next_key, next_key});
        for (int k = 0; k < kNumReads; ++k) {
          // kNumReads map reads.
          time_map.find(next_key);
          next_key = (next_key + kKeyStep) % kMaxKey;
        }
        // One map erase.
        auto iter = time_map.find(next_key);
        if (iter != time_map.end()) {
          time_map.erase(iter);
        }
      }
    });
  }

  pool.Schedule([=, &time_map]() {
    // Iterate the map entries while parallel inserts proceed.
    for (int i = 0; i < 1000; ++i) {
      time_map.insert({i, i});
    }
    for (auto it = time_map.begin(); it != time_map.end(); ++it) {
      it->second++;
    }
  });
}

// Measures the ellapsed time of a function invocation.
absl::Duration time(const std::function<void()>& f) {
  absl::Time start = absl::Now();
  f();
  return absl::Now() - start;
}

// Benchmarks a ShardedMap accessed by several parallel threads.
// With bazel build -c opt, the ShardedMap reduces CPU time by 60%.
TEST(ShardedMapTest, TestParallelAccess) {
  absl::Duration simple_time = time([] {
    absl::node_hash_map<int64, int64> simple_map;
    TestParallelAccess(simple_map, 1);
  });
  absl::Duration safe_time = time([] {
    ShardedMap<int64, int64> safe_map(4999, 1);
    TestParallelAccess(safe_map, 13);
  });
  absl::Duration sharded_time = time([] {
    ShardedMap<int64, int64> sharded_map(4999);
    TestParallelAccess(sharded_map, 13);
  });
  LOG(INFO) << "Ellapsed time: simple_map: " << simple_time;
  LOG(INFO) << "Ellapsed time: safe_map: " << safe_time;
  LOG(INFO) << "Ellapsed time: sharded_map: " << sharded_time;
}

}  // namespace
