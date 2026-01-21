// Copyright 2025 The MediaPipe Authors.
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

#include "mediapipe/framework/api3/internal/has_update_contract.h"

#include "absl/status/status.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3 {
namespace {

struct ContractType {};

struct WithUpdateContract {
  static absl::Status UpdateContract(ContractType&) {
    return absl::OkStatus();
  };
};

struct WithOutUpdateContract {};

TEST(HasUpdateContractTest, CanIdentifyUpdateContract) {
  EXPECT_TRUE((kHasUpdateContract<WithUpdateContract, ContractType>));
  EXPECT_FALSE((kHasUpdateContract<WithOutUpdateContract, ContractType>));
}

}  // namespace
}  // namespace mediapipe::api3
