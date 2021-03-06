// Copyright 2020 The MediaPipe Authors.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/point2.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Projects detections to a different coordinate system using a provided
// projection matrix.
//
// Input:
//   DETECTIONS - std::vector<Detection>
//     Detections to project using the provided projection matrix.
//   PROJECTION_MATRIX - std::array<float, 16>
//     A 4x4 row-major-order matrix that maps data from one coordinate system to
//     another.
//
// Output:
//   DETECTIONS - std::vector<Detection>
//     Projected detections.
//
// Example:
//   node {
//     calculator: "DetectionProjectionCalculator"
//     input_stream: "DETECTIONS:detections"
//     input_stream: "PROJECTION_MATRIX:matrix"
//     output_stream: "DETECTIONS:projected_detections"
//   }
class DetectionProjectionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(DetectionProjectionCalculator);

namespace {

constexpr char kDetections[] = "DETECTIONS";
constexpr char kProjectionMatrix[] = "PROJECTION_MATRIX";

absl::Status ProjectDetection(
    const std::function<Point2_f(const Point2_f&)>& project_fn,
    Detection* detection) {
  auto* location_data = detection->mutable_location_data();
  RET_CHECK_EQ(location_data->format(), LocationData::RELATIVE_BOUNDING_BOX);

  // Project keypoints.
  for (int i = 0; i < location_data->relative_keypoints_size(); ++i) {
    auto* kp = location_data->mutable_relative_keypoints(i);
    const auto point = project_fn({kp->x(), kp->y()});
    kp->set_x(point.x());
    kp->set_y(point.y());
  }

  // Project bounding box.
  auto* box = location_data->mutable_relative_bounding_box();

  const float xmin = box->xmin();
  const float ymin = box->ymin();
  const float width = box->width();
  const float height = box->height();
  // a) Define and project box points.
  std::array<Point2_f, 4> box_coordinates = {
      Point2_f{xmin, ymin}, Point2_f{xmin + width, ymin},
      Point2_f{xmin + width, ymin + height}, Point2_f{xmin, ymin + height}};
  std::transform(box_coordinates.begin(), box_coordinates.end(),
                 box_coordinates.begin(), project_fn);
  // b) Find new left top and right bottom points for a box which encompases
  //    non-projected (rotated) box.
  constexpr float kFloatMax = std::numeric_limits<float>::max();
  constexpr float kFloatMin = std::numeric_limits<float>::lowest();
  Point2_f left_top = {kFloatMax, kFloatMax};
  Point2_f right_bottom = {kFloatMin, kFloatMin};
  std::for_each(box_coordinates.begin(), box_coordinates.end(),
                [&left_top, &right_bottom](const Point2_f& p) {
                  left_top.set_x(std::min(left_top.x(), p.x()));
                  left_top.set_y(std::min(left_top.y(), p.y()));
                  right_bottom.set_x(std::max(right_bottom.x(), p.x()));
                  right_bottom.set_y(std::max(right_bottom.y(), p.y()));
                });
  box->set_xmin(left_top.x());
  box->set_ymin(left_top.y());
  box->set_width(right_bottom.x() - left_top.x());
  box->set_height(right_bottom.y() - left_top.y());

  return absl::OkStatus();
}

}  // namespace

absl::Status DetectionProjectionCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kDetections) &&
            cc->Inputs().HasTag(kProjectionMatrix))
      << "Missing one or more input streams.";

  RET_CHECK_EQ(cc->Inputs().NumEntries(kDetections),
               cc->Outputs().NumEntries(kDetections))
      << "Same number of DETECTIONS input and output is required.";

  for (CollectionItemId id = cc->Inputs().BeginId(kDetections);
       id != cc->Inputs().EndId(kDetections); ++id) {
    cc->Inputs().Get(id).Set<std::vector<Detection>>();
  }
  cc->Inputs().Tag(kProjectionMatrix).Set<std::array<float, 16>>();

  for (CollectionItemId id = cc->Outputs().BeginId(kDetections);
       id != cc->Outputs().EndId(kDetections); ++id) {
    cc->Outputs().Get(id).Set<std::vector<Detection>>();
  }

  return absl::OkStatus();
}

absl::Status DetectionProjectionCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status DetectionProjectionCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kProjectionMatrix).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& project_mat =
      cc->Inputs().Tag(kProjectionMatrix).Get<std::array<float, 16>>();
  auto project_fn = [project_mat](const Point2_f& p) -> Point2_f {
    return {p.x() * project_mat[0] + p.y() * project_mat[1] + project_mat[3],
            p.x() * project_mat[4] + p.y() * project_mat[5] + project_mat[7]};
  };

  CollectionItemId input_id = cc->Inputs().BeginId(kDetections);
  CollectionItemId output_id = cc->Outputs().BeginId(kDetections);
  // Number of inputs and outpus is the same according to the contract.
  for (; input_id != cc->Inputs().EndId(kDetections); ++input_id, ++output_id) {
    const auto& input_packet = cc->Inputs().Get(input_id);
    if (input_packet.IsEmpty()) {
      continue;
    }

    std::vector<Detection> output_detections;
    for (const auto& detection : input_packet.Get<std::vector<Detection>>()) {
      Detection output_detection = detection;
      MP_RETURN_IF_ERROR(ProjectDetection(project_fn, &output_detection));
      output_detections.push_back(std::move(output_detection));
    }

    cc->Outputs().Get(output_id).AddPacket(
        MakePacket<std::vector<Detection>>(std::move(output_detections))
            .At(cc->InputTimestamp()));
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
