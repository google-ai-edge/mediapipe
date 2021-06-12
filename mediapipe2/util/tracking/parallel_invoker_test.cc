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

#include "mediapipe/util/tracking/parallel_invoker.h"

#include <algorithm>
#include <numeric>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

void RunParallelTest() {
  absl::Mutex numbers_mutex;
  std::vector<int> numbers;
  const int kArraySize = 5000;

  // Fill number array in parallel.
  ParallelFor(0, kArraySize, 1,
              [&numbers_mutex, &numbers](const BlockedRange& b) {
                for (int k = b.begin(); k != b.end(); ++k) {
                  absl::MutexLock lock(&numbers_mutex);
                  numbers.push_back(k);
                }
              });

  std::vector<int> expected(kArraySize);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_TRUE(
      std::is_permutation(expected.begin(), expected.end(), numbers.begin()));
}

TEST(ParallelInvokerTest, PhotosTest) {
  flags_parallel_invoker_mode = PARALLEL_INVOKER_OPENMP;

  RunParallelTest();
}

TEST(ParallelInvokerTest, ThreadPoolTest) {
  flags_parallel_invoker_mode = PARALLEL_INVOKER_THREAD_POOL;

  // Needs to be run in opt mode to pass.
  RunParallelTest();
}

}  // namespace
}  // namespace mediapipe
