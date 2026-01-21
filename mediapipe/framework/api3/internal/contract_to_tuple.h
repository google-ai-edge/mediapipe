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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_TO_TUPLE_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_TO_TUPLE_H_

#include <any>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mediapipe/framework/api3/internal/dependent_false.h"

namespace mediapipe::api3 {

// Defined only if ContractT can be brace constructed from {Fs...}.
template <typename ContractT, typename... Fs>
decltype(ContractT{std::declval<Fs>()...},
         std::true_type{}) IsBraceConsructible(std::any);

// Variadic function template to fallback when ContractT cannot be constructed
// from {Fs...}.
template <typename ContractT, typename... Fs>
std::false_type IsBraceConsructible(...);

// Only intended to be used with MediaPipe interfaces as described in node.h.
template <typename ContractT, typename... Fs>
inline constexpr auto kIsBraceConstructible =
    decltype(IsBraceConsructible<ContractT, Fs...>(std::declval<std::any>())){};

class FieldPlaceholder {
 public:
  template <typename T>
  constexpr
  operator T();  // NOLINT: intentionally implicitly convertible and to be used
                 // only in conjunction with kIsBraceConstructible.

 private:
  FieldPlaceholder() = delete;
};

// FieldPlaceholders
#define INTERNAL_MEDIAPIPE_FP_1 FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_2 INTERNAL_MEDIAPIPE_FP_1, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_3 INTERNAL_MEDIAPIPE_FP_2, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_4 INTERNAL_MEDIAPIPE_FP_3, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_5 INTERNAL_MEDIAPIPE_FP_4, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_6 INTERNAL_MEDIAPIPE_FP_5, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_7 INTERNAL_MEDIAPIPE_FP_6, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_8 INTERNAL_MEDIAPIPE_FP_7, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_9 INTERNAL_MEDIAPIPE_FP_8, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_10 INTERNAL_MEDIAPIPE_FP_9, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_11 INTERNAL_MEDIAPIPE_FP_10, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_12 INTERNAL_MEDIAPIPE_FP_11, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_13 INTERNAL_MEDIAPIPE_FP_12, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_14 INTERNAL_MEDIAPIPE_FP_13, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_15 INTERNAL_MEDIAPIPE_FP_14, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_16 INTERNAL_MEDIAPIPE_FP_15, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_17 INTERNAL_MEDIAPIPE_FP_16, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_18 INTERNAL_MEDIAPIPE_FP_17, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_19 INTERNAL_MEDIAPIPE_FP_18, FieldPlaceholder
#define INTERNAL_MEDIAPIPE_FP_20 INTERNAL_MEDIAPIPE_FP_19, FieldPlaceholder

// Variable Names (f1, f2, ..., fN)
#define INTERNAL_MEDIAPIPE_VARS_1 f1
#define INTERNAL_MEDIAPIPE_VARS_2 INTERNAL_MEDIAPIPE_VARS_1, f2
#define INTERNAL_MEDIAPIPE_VARS_3 INTERNAL_MEDIAPIPE_VARS_2, f3
#define INTERNAL_MEDIAPIPE_VARS_4 INTERNAL_MEDIAPIPE_VARS_3, f4
#define INTERNAL_MEDIAPIPE_VARS_5 INTERNAL_MEDIAPIPE_VARS_4, f5
#define INTERNAL_MEDIAPIPE_VARS_6 INTERNAL_MEDIAPIPE_VARS_5, f6
#define INTERNAL_MEDIAPIPE_VARS_7 INTERNAL_MEDIAPIPE_VARS_6, f7
#define INTERNAL_MEDIAPIPE_VARS_8 INTERNAL_MEDIAPIPE_VARS_7, f8
#define INTERNAL_MEDIAPIPE_VARS_9 INTERNAL_MEDIAPIPE_VARS_8, f9
#define INTERNAL_MEDIAPIPE_VARS_10 INTERNAL_MEDIAPIPE_VARS_9, f10
#define INTERNAL_MEDIAPIPE_VARS_11 INTERNAL_MEDIAPIPE_VARS_10, f11
#define INTERNAL_MEDIAPIPE_VARS_12 INTERNAL_MEDIAPIPE_VARS_11, f12
#define INTERNAL_MEDIAPIPE_VARS_13 INTERNAL_MEDIAPIPE_VARS_12, f13
#define INTERNAL_MEDIAPIPE_VARS_14 INTERNAL_MEDIAPIPE_VARS_13, f14
#define INTERNAL_MEDIAPIPE_VARS_15 INTERNAL_MEDIAPIPE_VARS_14, f15
#define INTERNAL_MEDIAPIPE_VARS_16 INTERNAL_MEDIAPIPE_VARS_15, f16
#define INTERNAL_MEDIAPIPE_VARS_17 INTERNAL_MEDIAPIPE_VARS_16, f17
#define INTERNAL_MEDIAPIPE_VARS_18 INTERNAL_MEDIAPIPE_VARS_17, f18
#define INTERNAL_MEDIAPIPE_VARS_19 INTERNAL_MEDIAPIPE_VARS_18, f19
#define INTERNAL_MEDIAPIPE_VARS_20 INTERNAL_MEDIAPIPE_VARS_19, f20

// Address-Of Variables (&f1, &f2, ..., &fN)
#define INTERNAL_MEDIAPIPE_ADDR_VARS_1 &f1
#define INTERNAL_MEDIAPIPE_ADDR_VARS_2 INTERNAL_MEDIAPIPE_ADDR_VARS_1, &f2
#define INTERNAL_MEDIAPIPE_ADDR_VARS_3 INTERNAL_MEDIAPIPE_ADDR_VARS_2, &f3
#define INTERNAL_MEDIAPIPE_ADDR_VARS_4 INTERNAL_MEDIAPIPE_ADDR_VARS_3, &f4
#define INTERNAL_MEDIAPIPE_ADDR_VARS_5 INTERNAL_MEDIAPIPE_ADDR_VARS_4, &f5
#define INTERNAL_MEDIAPIPE_ADDR_VARS_6 INTERNAL_MEDIAPIPE_ADDR_VARS_5, &f6
#define INTERNAL_MEDIAPIPE_ADDR_VARS_7 INTERNAL_MEDIAPIPE_ADDR_VARS_6, &f7
#define INTERNAL_MEDIAPIPE_ADDR_VARS_8 INTERNAL_MEDIAPIPE_ADDR_VARS_7, &f8
#define INTERNAL_MEDIAPIPE_ADDR_VARS_9 INTERNAL_MEDIAPIPE_ADDR_VARS_8, &f9
#define INTERNAL_MEDIAPIPE_ADDR_VARS_10 INTERNAL_MEDIAPIPE_ADDR_VARS_9, &f10
#define INTERNAL_MEDIAPIPE_ADDR_VARS_11 INTERNAL_MEDIAPIPE_ADDR_VARS_10, &f11
#define INTERNAL_MEDIAPIPE_ADDR_VARS_12 INTERNAL_MEDIAPIPE_ADDR_VARS_11, &f12
#define INTERNAL_MEDIAPIPE_ADDR_VARS_13 INTERNAL_MEDIAPIPE_ADDR_VARS_12, &f13
#define INTERNAL_MEDIAPIPE_ADDR_VARS_14 INTERNAL_MEDIAPIPE_ADDR_VARS_13, &f14
#define INTERNAL_MEDIAPIPE_ADDR_VARS_15 INTERNAL_MEDIAPIPE_ADDR_VARS_14, &f15
#define INTERNAL_MEDIAPIPE_ADDR_VARS_16 INTERNAL_MEDIAPIPE_ADDR_VARS_15, &f16
#define INTERNAL_MEDIAPIPE_ADDR_VARS_17 INTERNAL_MEDIAPIPE_ADDR_VARS_16, &f17
#define INTERNAL_MEDIAPIPE_ADDR_VARS_18 INTERNAL_MEDIAPIPE_ADDR_VARS_17, &f18
#define INTERNAL_MEDIAPIPE_ADDR_VARS_19 INTERNAL_MEDIAPIPE_ADDR_VARS_18, &f19
#define INTERNAL_MEDIAPIPE_ADDR_VARS_20 INTERNAL_MEDIAPIPE_ADDR_VARS_19, &f20

#define INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(N)   \
  (kIsBraceConstructible<ContractT, INTERNAL_MEDIAPIPE_FP_##N>) { \
    auto& [INTERNAL_MEDIAPIPE_VARS_##N] = contract;               \
    return std::make_tuple(INTERNAL_MEDIAPIPE_ADDR_VARS_##N);     \
  }

// Function takes a struct (which represents a specialized contract as
// described in contract.h) and returns `std::tuple` of pointers to every field
// of the struct.
//
// E.g.
// ```
//   template <typename S>
//   struct Foo {
//     Input<S, int> in{"IN"};
//     Output<S, std::string> out{"OUT"};
//   };
// ```
// For `Foo<ContractSpecializer> contract;` it will return:
// ```
//   std::tuple<Input<ContractSpecializer, int>*,
//              Output<ContractSpecializer, std::string>*>`
// ```
// object which enables alternative access to structure fields: e.g. iterating
// and performing extra initialization when required.
//
// How this works at compile time:
// 1. Check if a passed structure brace constructible from particular number of
//    arguments: (`contract = {placeholder, placeholder};`)
// 2. Use structured binding to get/list all fields.
//    (`auto& [arg1, arg2] = contract;`)
// 3. Return tuple of pointers.
//    (`return std::make_tuple(&arg1, &arg2)`)
//
// Currently supporting only contract/structure with 20 fields.
//
// NOTE: It's fine to increase the number of ports as needed. However, before
// doing so, it should be considered whether it's the right choice to have so
// many ports in the calculator - it's like having at lest 9 params in a
// function - which might be worth to consider if splitting into more than one
// calculator can do the job or introducing dedicated aggregate types for your
// inputs/outputs.
template <typename ContractT>
auto ContractToFieldPtrTuple(ContractT& contract) {
  if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(20)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(19)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(18)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(17)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(16)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(15)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(14)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(13)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(12)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(11)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(10)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(9)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(8)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(7)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(6)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(5)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(4)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(3)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(2)
  else if constexpr
    INTERNAL_MEDIAPIPE_GENERATE_BRACE_CONSTRUCTIBLE_CASE(1)
  else if constexpr (kIsBraceConstructible<ContractT>)
    return std::make_tuple();
  else
    static_assert(
        dependent_false_v<ContractT>,
        "Unsupported contract: the current limit is 20 fields per contract.");
}

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_TO_TUPLE_H_
