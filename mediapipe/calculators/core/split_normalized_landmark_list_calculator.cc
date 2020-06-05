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

#ifndef MEDIAPIPE_CALCULATORS_CORE_SPLIT_NORMALIZED_LANDMARK_LIST_CALCULATOR_H_  // NOLINT
#define MEDIAPIPE_CALCULATORS_CORE_SPLIT_NORMALIZED_LANDMARK_LIST_CALCULATOR_H_  // NOLINT

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

// Splits an input packet with NormalizedLandmarkList into
// multiple NormalizedLandmarkList output packets using the [begin, end) ranges
// specified in SplitVectorCalculatorOptions. If the option "element_only" is
// set to true, all ranges should be of size 1 and all outputs will be elements
// of type NormalizedLandmark. If "element_only" is false, ranges can be
// non-zero in size and all outputs will be of type NormalizedLandmarkList.
// If the option "combine_outputs" is set to true, only one output stream can be
// specified and all ranges of elements will be combined into one
// NormalizedLandmarkList.
class SplitNormalizedLandmarkListCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().NumEntries() == 1);
    RET_CHECK(cc->Outputs().NumEntries() != 0);

    cc->Inputs().Index(0).Set<NormalizedLandmarkList>();

    const auto& options =
        cc->Options<::mediapipe::SplitVectorCalculatorOptions>();

    if (options.combine_outputs()) {
      RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
      cc->Outputs().Index(0).Set<NormalizedLandmarkList>();
      for (int i = 0; i < options.ranges_size() - 1; ++i) {
        for (int j = i + 1; j < options.ranges_size(); ++j) {
          const auto& range_0 = options.ranges(i);
          const auto& range_1 = options.ranges(j);
          if ((range_0.begin() >= range_1.begin() &&
               range_0.begin() < range_1.end()) ||
              (range_1.begin() >= range_0.begin() &&
               range_1.begin() < range_0.end())) {
            return ::mediapipe::InvalidArgumentError(
                "Ranges must be non-overlapping when using combine_outputs "
                "option.");
          }
        }
      }
    } else {
      if (cc->Outputs().NumEntries() != options.ranges_size()) {
        return ::mediapipe::InvalidArgumentError(
            "The number of output streams should match the number of ranges "
            "specified in the CalculatorOptions.");
      }

      // Set the output types for each output stream.
      for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
        if (options.ranges(i).begin() < 0 || options.ranges(i).end() < 0 ||
            options.ranges(i).begin() >= options.ranges(i).end()) {
          return ::mediapipe::InvalidArgumentError(
              "Indices should be non-negative and begin index should be less "
              "than the end index.");
        }
        if (options.element_only()) {
          if (options.ranges(i).end() - options.ranges(i).begin() != 1) {
            return ::mediapipe::InvalidArgumentError(
                "Since element_only is true, all ranges should be of size 1.");
          }
          cc->Outputs().Index(i).Set<NormalizedLandmark>();
        } else {
          cc->Outputs().Index(i).Set<NormalizedLandmarkList>();
        }
      }
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    const auto& options =
        cc->Options<::mediapipe::SplitVectorCalculatorOptions>();

    element_only_ = options.element_only();
    combine_outputs_ = options.combine_outputs();

    for (const auto& range : options.ranges()) {
      ranges_.push_back({range.begin(), range.end()});
      max_range_end_ = std::max(max_range_end_, range.end());
      total_elements_ += range.end() - range.begin();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    const NormalizedLandmarkList& input =
        cc->Inputs().Index(0).Get<NormalizedLandmarkList>();
    RET_CHECK_GE(input.landmark_size(), max_range_end_)
        << "Max range end " << max_range_end_ << " exceeds landmarks size "
        << input.landmark_size();

    if (combine_outputs_) {
      NormalizedLandmarkList output;
      for (int i = 0; i < ranges_.size(); ++i) {
        for (int j = ranges_[i].first; j < ranges_[i].second; ++j) {
          const NormalizedLandmark& input_landmark = input.landmark(j);
          *output.add_landmark() = input_landmark;
        }
      }
      RET_CHECK_EQ(output.landmark_size(), total_elements_);
      cc->Outputs().Index(0).AddPacket(
          MakePacket<NormalizedLandmarkList>(output).At(cc->InputTimestamp()));
    } else {
      if (element_only_) {
        for (int i = 0; i < ranges_.size(); ++i) {
          cc->Outputs().Index(i).AddPacket(
              MakePacket<NormalizedLandmark>(input.landmark(ranges_[i].first))
                  .At(cc->InputTimestamp()));
        }
      } else {
        for (int i = 0; i < ranges_.size(); ++i) {
          NormalizedLandmarkList output;
          for (int j = ranges_[i].first; j < ranges_[i].second; ++j) {
            const NormalizedLandmark& input_landmark = input.landmark(j);
            *output.add_landmark() = input_landmark;
          }
          cc->Outputs().Index(i).AddPacket(
              MakePacket<NormalizedLandmarkList>(output).At(
                  cc->InputTimestamp()));
        }
      }
    }

    return ::mediapipe::OkStatus();
  }

 private:
  std::vector<std::pair<int32, int32>> ranges_;
  int32 max_range_end_ = -1;
  int32 total_elements_ = 0;
  bool element_only_ = false;
  bool combine_outputs_ = false;
};

REGISTER_CALCULATOR(SplitNormalizedLandmarkListCalculator);

}  // namespace mediapipe

// NOLINTNEXTLINE
#endif  // MEDIAPIPE_CALCULATORS_CORE_SPLIT_NORMALIZED_LANDMARK_LIST_CALCULATOR_H_
