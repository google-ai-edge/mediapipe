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

#ifndef MEDIAPIPE_FRAMEWORK_TEST_SERVICE_H_
#define MEDIAPIPE_FRAMEWORK_TEST_SERVICE_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"

namespace mediapipe {

using TestServiceObject = std::map<std::string, int>;

inline constexpr GraphService<TestServiceObject> kTestService(
    "test_service", GraphServiceBase::kDisallowDefaultInitialization);
inline constexpr GraphService<int> kAnotherService(
    "another_service", GraphServiceBase::kAllowDefaultInitialization);

class NoDefaultConstructor {
 public:
  NoDefaultConstructor() = delete;
};
inline constexpr GraphService<NoDefaultConstructor> kNoDefaultService(
    "no_default_service", GraphServiceBase::kAllowDefaultInitialization);

class NeedsCreateMethod {
 public:
  static absl::StatusOr<std::shared_ptr<NeedsCreateMethod>> Create() {
    return std::shared_ptr<NeedsCreateMethod>(new NeedsCreateMethod());
  }

 private:
  NeedsCreateMethod() = default;
};
inline constexpr GraphService<NeedsCreateMethod> kNeedsCreateService(
    "needs_create_service", GraphServiceBase::kAllowDefaultInitialization);

// Use a service.
class TestServiceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) final;
  absl::Status Process(CalculatorContext* cc) final;

 private:
  int optional_bias_ = 0;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TEST_SERVICE_H_
