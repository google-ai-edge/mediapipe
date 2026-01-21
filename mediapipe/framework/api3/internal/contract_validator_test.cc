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

#include "mediapipe/framework/api3/internal/contract_validator.h"

#include <string>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/testing/foo.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {

template <typename S>
struct Foo {
  Input<S, int> in{"IN"};
  Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
  Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};

  SideInput<S, std::string> side_in{"SIDE_IN"};
  Optional<SideInput<S, std::string>> optional_side_in{"OPTIONAL_SIDE_IN"};
  Repeated<SideInput<S, std::string>> repeated_side_in{"REPEATED_SIDE_IN"};

  Output<S, int> out{"OUT"};
  Optional<Output<S, int>> optional_out{"OPTIONAL_OUT"};
  Repeated<Output<S, int>> repeated_out{"REPEATED_OUT"};

  SideOutput<S, std::string> side_out{"SIDE_OUT"};
  Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};
  Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};

  Options<S, FooOptions> options;
};

TEST(ContractValidatorTest, PassesForCorrectContract) {
  absl::Status status = ValidateContract<Foo>();
  MP_EXPECT_OK(status);
}

template <typename S>
struct SomeContractWithDuplicateInputs {
  Input<S, int> in_a{"IN"};
  Input<S, int> in_b{"IN"};
  Output<S, int> out{"OUT"};
};

TEST(ContractValidatorTest, FailsWithDuplicateInputs) {
  absl::Status status = ValidateContract<SomeContractWithDuplicateInputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeContractWithDuplicateOutputs {
  Input<S, int> in{"IN"};
  Output<S, int> out_a{"OUT"};
  Output<S, int> out_b{"OUT"};
};

TEST(ContractValidatorTest, FailsWithDuplicateOutputs) {
  absl::Status status = ValidateContract<SomeContractWithDuplicateOutputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeContractWithDuplicateSideInputs {
  SideInput<S, int> side_in_a{"IN"};
  SideInput<S, int> side_in_b{"IN"};
};

TEST(ContractValidatorTest, FailsWithDuplicateSideInputs) {
  absl::Status status = ValidateContract<SomeContractWithDuplicateSideInputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeContractWithDuplicateSideOutputs {
  SideOutput<S, int> side_out_a{"OUT"};
  SideOutput<S, int> side_out_b{"OUT"};
};

TEST(ContractValidatorTest, FailsWithDuplicateSideOutputs) {
  absl::Status status =
      ValidateContract<SomeContractWithDuplicateSideOutputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeIncorrectContractWithRepeatedInputs {
  Repeated<Input<S, int>> repeated_in{"IN"};
  Input<S, int> in{"IN"};
};

TEST(ContractValidatorTest, FailsWithIncorrectRepeatedInputFields) {
  absl::Status status =
      ValidateContract<SomeIncorrectContractWithRepeatedInputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeIncorrectContractWithRepeatedOutputs {
  Repeated<Output<S, int>> repeated_out{"OUT"};
  Output<S, int> out{"OUT"};
};

TEST(ContractValidatorTest, FailsWithIncorrectRepeatedOutputFields) {
  absl::Status status =
      ValidateContract<SomeIncorrectContractWithRepeatedOutputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeIncorrectContractWithOptionalInputs {
  Optional<Input<S, int>> optional_in{"IN"};
  Input<S, int> in{"IN"};
};

TEST(ContractValidatorTest, FailsWithIncorrectOptionalInputFields) {
  absl::Status status =
      ValidateContract<SomeIncorrectContractWithOptionalInputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct SomeIncorrectContractWithOptionalOutputs {
  Optional<Output<S, int>> optional_out{"OUT"};
  Output<S, int> out{"OUT"};
};

TEST(ContractValidatorTest, FailsWithIncorrectOptionalOutputFields) {
  absl::Status status =
      ValidateContract<SomeIncorrectContractWithOptionalOutputs>();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAlreadyExists));
}

template <typename S>
struct MultipleDuplicateTagsContract {
  Input<S, int> in{"IN"};
  Optional<Input<S, int>> optional_in{"IN"};
  Repeated<Input<S, int>> repeated_in{"IN"};

  SideInput<S, std::string> side_in{"SIDE_IN"};
  Optional<SideInput<S, std::string>> optional_side_in{"SIDE_IN"};
  Repeated<SideInput<S, std::string>> repeated_side_in{"SIDE_IN"};

  Output<S, int> out{"OUT"};
  Optional<Output<S, int>> optional_out{"OUT"};
  Repeated<Output<S, int>> repeated_out{"OUT"};

  SideOutput<S, std::string> side_out{"SIDE_OUT"};
  Optional<SideOutput<S, std::string>> optional_side_out{"SIDE_OUT"};
  Repeated<SideOutput<S, std::string>> repeated_side_out{"SIDE_OUT"};

  Options<S, FooOptions> options;
};

TEST(ContractValidatorTest, FailsWithMultipleDuplicateTags) {
  absl::Status status = ValidateContract<MultipleDuplicateTagsContract>();
  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kAlreadyExists,
               testing::HasSubstr(
                   "Field at index [11] has a duplicate tag: SIDE_OUT")));
}

TEST(ContractValidatorTest, TerminatesOnInvalidContract) {
  EXPECT_DEATH(IsContractValid<MultipleDuplicateTagsContract>(),
               testing::HasSubstr("is invalid"));
}

}  // namespace mediapipe::api3
