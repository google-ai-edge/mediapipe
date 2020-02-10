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

#include "mediapipe/framework/counter_factory.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace mediapipe {
namespace {

// Counter implementation when we're not using Flume.
// TODO: Consider using Dax atomic counters instead of this.
// This class is thread safe.
class BasicCounter : public Counter {
 public:
  explicit BasicCounter(const std::string& name) : value_(0) {}

  void Increment() ABSL_LOCKS_EXCLUDED(mu_) override {
    absl::WriterMutexLock lock(&mu_);
    ++value_;
  }

  void IncrementBy(int amount) ABSL_LOCKS_EXCLUDED(mu_) override {
    absl::WriterMutexLock lock(&mu_);
    value_ += amount;
  }

  int64 Get() ABSL_LOCKS_EXCLUDED(mu_) override {
    absl::ReaderMutexLock lock(&mu_);
    return value_;
  }

 private:
  absl::Mutex mu_;
  int64 value_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

CounterSet::CounterSet() {}

CounterSet::~CounterSet() ABSL_LOCKS_EXCLUDED(mu_) { PublishCounters(); }

void CounterSet::PublishCounters() ABSL_LOCKS_EXCLUDED(mu_) {}

void CounterSet::PrintCounters() ABSL_LOCKS_EXCLUDED(mu_) {
  absl::ReaderMutexLock lock(&mu_);
  LOG_IF(INFO, !counters_.empty()) << "MediaPipe Counters:";
  for (const auto& counter : counters_) {
    LOG(INFO) << counter.first << ": " << counter.second->Get();
  }
}

Counter* CounterSet::Get(const std::string& name) ABSL_LOCKS_EXCLUDED(mu_) {
  absl::ReaderMutexLock lock(&mu_);
  if (!::mediapipe::ContainsKey(counters_, name)) {
    return nullptr;
  }
  return counters_[name].get();
}

std::map<std::string, int64> CounterSet::GetCountersValues()
    ABSL_LOCKS_EXCLUDED(mu_) {
  absl::ReaderMutexLock lock(&mu_);
  std::map<std::string, int64> result;
  for (const auto& it : counters_) {
    result[it.first] = it.second->Get();
  }
  return result;
}

Counter* BasicCounterFactory::GetCounter(const std::string& name) {
  return counter_set_.Emplace<BasicCounter>(name, name);
}

}  // namespace mediapipe
