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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_CONTRACT_VALIDATOR_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_CONTRACT_VALIDATOR_H_

#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/contract_to_tuple.h"
#include "mediapipe/framework/api3/internal/dependent_false.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/deps/registration.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe::api3 {

// Contract validator is used internally to verify that contract is defined
// correctly - unique tags for fields of the same kind (e.g. Input) and
// `Repeated`/`Optional` of the same kind.
//
// Example of invalid contract:
//
// ```
//   template <typename S>
//   struct InvalidFoo {
//     Input<S, int> in{"IN"};
//     Optional<Input<S, float>> optional_in{"IN"};
//     Repeated<Input<S, double>> repeated_in{"IN"};
//   };
// ```
//
// "IN" is used as a tag for all inputs - MediaPipe has no way to handle this.
// Also, in this case it's tricky because the second input is an optional
// connection, and the issue (crash) may strike you only in production on some
// condition - validator, helps to ensure invalid contract is identified sooner
// than later.
//
// Check only happens on DEBUG builds when contract is used: e.g.
// Calculator<YourNode, ...> - that triggers static initialization of a contract
// static variable which results in a crash if contract for YourNode is invalid.
//
// Why only DEBUG builds:
// - validator utilizes static initialization to do validation for contract
//   uncoditionally and extra static intialization is unwanted
// - you must have tests and they run in DEBUG and it's OK to have some extra
//   static initialization there
template <template <typename, typename...> typename ContractT, typename... Ts>
class ContractValidator;

struct ContractInfo {
  absl::flat_hash_set<std::string> input_tags;
  absl::flat_hash_set<std::string> output_tags;
  absl::flat_hash_set<std::string> side_input_tags;
  absl::flat_hash_set<std::string> side_output_tags;
};

template <typename FieldT>
absl::Status ValidateField(absl::string_view tag, int field_index,
                           ContractInfo& info) {
  auto& tags = [&info]() -> absl::flat_hash_set<std::string>& {
    if constexpr (std::is_same_v<FieldT, InputStreamField>) {
      return info.input_tags;
    } else if constexpr (std::is_same_v<FieldT, OutputStreamField>) {
      return info.output_tags;
    } else if constexpr (std::is_same_v<FieldT, InputSidePacketField>) {
      return info.side_input_tags;
    } else if constexpr (std::is_same_v<FieldT, OutputSidePacketField>) {
      return info.side_output_tags;
    } else {
      static_assert(dependent_false<FieldT>::value, "Unexpected field type.");
    }
  }();

  auto [it, inserted] = tags.insert(std::string(tag));
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Field at index [", field_index, "] has a duplicate tag: ", tag));
  }
  return absl::OkStatus();
}

// Validates the provided contract.
// - old distinct field types must have distinct tags.
template <template <typename, typename...> typename ContractT, typename... Ts>
absl::Status ValidateContract() {
  ContractT<ContractSpecializer, Ts...> contract;
  auto ptrs = ContractToFieldPtrTuple(contract);
  ContractInfo contract_info;
  std::vector<absl::Status> errors;
  int field_index = 0;
  std::apply(
      [&](auto&... field) {
        (([&] {
           using FieldT = typename std::decay_t<decltype(*field)>::Field;
           absl::Status status;
           if constexpr (std::is_same_v<FieldT, RepeatedField> ||
                         std::is_same_v<FieldT, OptionalField>) {
             using ContainedField =
                 typename std::decay_t<decltype(*field)>::Contained::Field;
             status = ValidateField<ContainedField>(field->Tag(), field_index,
                                                    contract_info);
           } else if constexpr (std::is_same_v<FieldT, OptionsField>) {
             // Ignore Options as they are not ports, don't have a tag and
             // arbitrary number is allowed (usually just one).
           } else {
             status = ValidateField<FieldT>(field->Tag(), field_index,
                                            contract_info);
           }
           ++field_index;
           if (!status.ok()) {
             errors.push_back(status);
           }
         }()),
         ...);
      },
      ptrs);

  if (errors.empty()) return absl::OkStatus();
  return tool::CombinedStatus(
      absl::StrCat("Contract ",
                   typeid(ContractT<ContractSpecializer, Ts...>).name(),
                   " is invalid."),
      errors);
}

template <template <typename, typename...> typename ContractT, typename... Ts>
bool IsContractValid() {
  absl::Status status = ValidateContract<ContractT, Ts...>();
  if (status.ok()) return true;
  ABSL_LOG(FATAL) << status;
}

#ifdef NDEBUG

template <template <typename, typename...> typename ContractT, typename... Ts>
class ContractValidator {};

#else

template <template <typename, typename...> typename ContractT, typename... Ts>
class ContractValidator {
 private:
  inline static const bool is_valid_ = IsContractValid<ContractT, Ts...>();

  using RequireStatics =
      registration_internal::ForceStaticInstantiation<&is_valid_>;
};

#endif

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_CONTRACT_VALIDATOR_H_
