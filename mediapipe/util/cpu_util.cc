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

#include "mediapipe/util/cpu_util.h"

#include <cmath>
#include <cstdint>

#ifdef __ANDROID__
#include "ndk/sources/android/cpufeatures/cpu-features.h"
#elif _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <fstream>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {
namespace {

constexpr uint32_t kBufferLength = 64;

absl::StatusOr<std::string> GetFilePath(int cpu) {
  return absl::Substitute(
      "/sys/devices/system/cpu/cpu$0/cpufreq/cpuinfo_max_freq", cpu);
}

absl::StatusOr<uint64_t> GetCpuMaxFrequency(int cpu) {
  auto path_or_status = GetFilePath(cpu);
  if (!path_or_status.ok()) {
    return path_or_status.status();
  }
  std::ifstream file;
  file.open(path_or_status.value());
  if (file.is_open()) {
    char buffer[kBufferLength];
    file.getline(buffer, kBufferLength);
    file.close();
    uint64_t frequency;
    if (absl::SimpleAtoi(buffer, &frequency)) {
      return frequency;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid frequency: ", buffer));
    }
  } else {
    return absl::NotFoundError(
        absl::StrCat("Couldn't read ", path_or_status.value()));
  }
}

std::set<int> InferLowerOrHigherCoreIds(bool lower) {
  std::vector<std::pair<int, uint64_t>> cpu_freq_pairs;
  for (int cpu = 0; cpu < NumCPUCores(); ++cpu) {
    auto freq_or_status = GetCpuMaxFrequency(cpu);
    if (freq_or_status.ok()) {
      cpu_freq_pairs.push_back({cpu, freq_or_status.value()});
    }
  }
  if (cpu_freq_pairs.empty()) {
    return {};
  }

  absl::c_sort(cpu_freq_pairs, [lower](const std::pair<int, uint64_t>& left,
                                       const std::pair<int, uint64_t>& right) {
    return (lower && left.second < right.second) ||
           (!lower && left.second > right.second);
  });
  uint64_t edge_freq = cpu_freq_pairs[0].second;

  std::set<int> inferred_cores;
  for (const auto& cpu_freq_pair : cpu_freq_pairs) {
    if ((lower && cpu_freq_pair.second > edge_freq) ||
        (!lower && cpu_freq_pair.second < edge_freq)) {
      break;
    }
    inferred_cores.insert(cpu_freq_pair.first);
  }

  // If all the cores have the same frequency, there are no "lower" or "higher"
  // cores.
  if (inferred_cores.size() == cpu_freq_pairs.size()) {
    return {};
  } else {
    return inferred_cores;
  }
}
}  // namespace

int NumCPUCores() {
#ifdef __ANDROID__
  return android_getCpuCount();
#elif _WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#else
  return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

std::set<int> InferLowerCoreIds() {
  return InferLowerOrHigherCoreIds(/* lower= */ true);
}

std::set<int> InferHigherCoreIds() {
  return InferLowerOrHigherCoreIds(/* lower= */ false);
}

}  // namespace mediapipe.
