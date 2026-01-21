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

#ifndef MEDIAPIPE_FRAMEWORK_API3_ANY_H_
#define MEDIAPIPE_FRAMEWORK_API3_ANY_H_

namespace mediapipe::api3 {

// `Any` type can be handy in the following use cases:
//
// 1. You have a node where you need to trigger execution, but input stream
//    trigerring the execution is not used in any other way.
// 2. You have a node which doesn't read packets or creates packets, but rather
//    passes them or drops thems (E.g. pass through calculators, gate
//    calculators).
//
// NOTE: in pass through cases, it's important to indicate that output, even
//   though it's `Any` is going to be the same actual type as the corresponding
//   actual input type.
//
//   To indicate this you need to use `SameAs` as following in `Contract` or
//   node `Calculator` implementation:
//   ```
//     static absl::Status UpdateContract(CalculatorContract<...>& cc) {
//       cc.output.SetSameAs(cc.input);
//       return absl::OkStatus();
//     }
//   ```
//   With repeated inputs/outputs, it's the same per input/output:
//   ```
//     static absl::Status UpdateContract(CalculatorContract<...>& cc) {
//       RET_CHECK_EQ(cc.out.Count(), cc.in.Count());
//       for (int i = 0; i < cc.out.Count(); ++i) {
//         cc.out.At(i).SetSameAs(cc.in.At(i));
//       }
//       return absl::OkStatus();
//     }
//   ```
//
// RECOMMENDATION: prefer templates for node contract definition where possible.
struct Any {};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_ANY_H_
