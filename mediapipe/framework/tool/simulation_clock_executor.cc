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

#include "mediapipe/framework/tool/simulation_clock_executor.h"

#include "mediapipe/framework/tool/simulation_clock.h"

namespace mediapipe {

SimulationClockExecutor::SimulationClockExecutor(int num_threads)
    : ThreadPoolExecutor(num_threads), clock_(new SimulationClock()) {}

void SimulationClockExecutor::Schedule(std::function<void()> task) {
  clock_->ThreadStart();
  ThreadPoolExecutor::Schedule([this, task] {
    clock_->Sleep(absl::ZeroDuration());
    task();
    clock_->ThreadFinish();
  });
}

std::shared_ptr<SimulationClock> SimulationClockExecutor::GetClock() {
  return clock_;
}

}  // namespace mediapipe
