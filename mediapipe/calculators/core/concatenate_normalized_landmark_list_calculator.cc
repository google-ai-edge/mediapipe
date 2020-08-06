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

#ifndef MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_NORMALIZED_LIST_CALCULATOR_H_  // NOLINT
#define MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_NORMALIZED_LIST_CALCULATOR_H_  // NOLINT

#include "mediapipe/calculators/core/concatenate_vector_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Concatenates several NormalizedLandmarkList protos following stream index
// order. This class assumes that every input stream contains a
// NormalizedLandmarkList proto object.
class ConcatenateNormalizedLandmarkListCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().NumEntries() != 0);
    RET_CHECK(cc->Outputs().NumEntries() == 1);

    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<NormalizedLandmarkList>();
    }

    cc->Outputs().Index(0).Set<NormalizedLandmarkList>();

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

    NormalizedLandmarkList output;
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      if (cc->Inputs().Index(i).IsEmpty()) continue;
      const NormalizedLandmarkList& input =
          cc->Inputs().Index(i).Get<NormalizedLandmarkList>();
      for (int j = 0; j < input.landmark_size(); ++j) {
        const NormalizedLandmark& input_landmark = input.landmark(j);
        *output.add_landmark() = input_landmark;
      }
    }
    cc->Outputs().Index(0).AddPacket(
        MakePacket<NormalizedLandmarkList>(output).At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
  }

 private:
  bool only_emit_if_all_present_;
};

REGISTER_CALCULATOR(ConcatenateNormalizedLandmarkListCalculator);

}  // namespace mediapipe

// NOLINTNEXTLINE
#endif  // MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_NORMALIZED_LIST_CALCULATOR_H_
