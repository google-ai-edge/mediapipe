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

#include "mediapipe/framework/deps/threadpool.h"

#include <set>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

TEST(ThreadPoolTest, DestroyWithoutStart) {
  ThreadPool thread_pool("testpool", 10);
}

TEST(ThreadPoolTest, EmptyThread) {
  ThreadPool thread_pool("testpool", 0);
  ASSERT_EQ(1, thread_pool.num_threads());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, SingleThread) {
  absl::Mutex mu;
  int n = 100;
  {
    ThreadPool thread_pool("testpool", 1);
    ASSERT_EQ(1, thread_pool.num_threads());
    thread_pool.StartWorkers();

    for (int i = 0; i < 100; ++i) {
      thread_pool.Schedule([&n, &mu]() mutable {
        absl::MutexLock l(&mu);
        --n;
      });
    }
  }

  EXPECT_EQ(0, n);
}

TEST(ThreadPoolTest, MultiThreads) {
  absl::Mutex mu;
  int n = 100;
  {
    ThreadPool thread_pool("testpool", 10);
    ASSERT_EQ(10, thread_pool.num_threads());
    thread_pool.StartWorkers();

    for (int i = 0; i < 100; ++i) {
      thread_pool.Schedule([&n, &mu]() mutable {
        absl::MutexLock l(&mu);
        --n;
      });
    }
  }

  EXPECT_EQ(0, n);
}

TEST(ThreadPoolTest, CreateWithThreadOptions) {
  ThreadPool thread_pool(ThreadOptions(), "testpool", 10);
  ASSERT_EQ(10, thread_pool.num_threads());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, CreateWithThreadPriority) {
  ThreadOptions thread_options = ThreadOptions().set_nice_priority_level(-10);
  ThreadPool thread_pool(thread_options, "testpool", 10);
  ASSERT_EQ(10, thread_pool.num_threads());
  ASSERT_EQ(-10, thread_pool.thread_options().nice_priority_level());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, CreateWithCPUAffinity) {
  ThreadOptions thread_options = ThreadOptions().set_cpu_set({0});
  ThreadPool thread_pool(thread_options, "testpool", 10);
  ASSERT_EQ(10, thread_pool.num_threads());
  ASSERT_EQ(1, thread_pool.thread_options().cpu_set().size());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, CreateThreadName) {
  ASSERT_EQ("name_prefix/123", internal::CreateThreadName("name_prefix", 1234));
  ASSERT_EQ("name_prefix/123",
            internal::CreateThreadName("name_prefix", 12345));
  ASSERT_EQ("name_prefix/123",
            internal::CreateThreadName("name_prefix", 123456));
  ASSERT_EQ("name_prefix/123",
            internal::CreateThreadName("name_prefix", 1234567));
  ASSERT_EQ("name_prefix/123",
            internal::CreateThreadName("name_prefix", 1234567891));
  ASSERT_EQ("name_prefix_/12",
            internal::CreateThreadName("name_prefix_", 1234));
  ASSERT_EQ("name_pre/123456",
            internal::CreateThreadName("name_pre", 1234567891));
  ASSERT_EQ("n/1", internal::CreateThreadName("n", 1));
  ASSERT_EQ("name_p/12345678",
            internal::CreateThreadName("name_p", 1234567891));
  ASSERT_EQ("/1", internal::CreateThreadName("", 1));
  ASSERT_EQ("name_prefix_lon",
            internal::CreateThreadName("name_prefix_long", 1234));
  ASSERT_EQ("name_prefix_lon",
            internal::CreateThreadName("name_prefix_lon", 1234));
}

}  // namespace mediapipe
