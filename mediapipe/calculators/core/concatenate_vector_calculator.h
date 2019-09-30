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

#ifndef MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_VECTOR_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_VECTOR_CALCULATOR_H_

#include <type_traits>
#include <vector>

#include "mediapipe/calculators/core/concatenate_vector_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Concatenates several std::vector<T> following stream index order. This class
// assumes that every input stream contains the vector<T> type. To use this
// class for a particular type T, regisiter a calculator using
// ConcatenateVectorCalculator<T>.
template <typename T>
class ConcatenateVectorCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().NumEntries() != 0);
    RET_CHECK(cc->Outputs().NumEntries() == 1);

    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<std::vector<T>>();
    }

    cc->Outputs().Index(0).Set<std::vector<T>>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    only_emit_if_all_present_ =
        cc->Options<::mediapipe::ConcatenateVectorCalculatorOptions>()
            .only_emit_if_all_present();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (only_emit_if_all_present_) {
      for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
        if (cc->Inputs().Index(i).IsEmpty()) return ::mediapipe::OkStatus();
      }
    }

    return ConcatenateVectors<T>(std::is_copy_constructible<T>(), cc);
  }

  template <typename U>
  ::mediapipe::Status ConcatenateVectors(std::true_type,
                                         CalculatorContext* cc) {
    auto output = absl::make_unique<std::vector<U>>();
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      if (cc->Inputs().Index(i).IsEmpty()) continue;
      const std::vector<U>& input = cc->Inputs().Index(i).Get<std::vector<U>>();
      output->insert(output->end(), input.begin(), input.end());
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }

  template <typename U>
  ::mediapipe::Status ConcatenateVectors(std::false_type,
                                         CalculatorContext* cc) {
    return ConsumeAndConcatenateVectors<T>(std::is_move_constructible<U>(), cc);
  }

  template <typename U>
  ::mediapipe::Status ConsumeAndConcatenateVectors(std::true_type,
                                                   CalculatorContext* cc) {
    auto output = absl::make_unique<std::vector<U>>();
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      if (cc->Inputs().Index(i).IsEmpty()) continue;
      ::mediapipe::StatusOr<std::unique_ptr<std::vector<U>>> input_status =
          cc->Inputs().Index(i).Value().Consume<std::vector<U>>();
      if (input_status.ok()) {
        std::unique_ptr<std::vector<U>> input_vector =
            std::move(input_status).ValueOrDie();
        output->insert(output->end(),
                       std::make_move_iterator(input_vector->begin()),
                       std::make_move_iterator(input_vector->end()));
      } else {
        return input_status.status();
      }
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }

  template <typename U>
  ::mediapipe::Status ConsumeAndConcatenateVectors(std::false_type,
                                                   CalculatorContext* cc) {
    return ::mediapipe::InternalError(
        "Cannot copy or move input vectors to concatenate them");
  }

 private:
  bool only_emit_if_all_present_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_VECTOR_CALCULATOR_H_
