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

#include <memory>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/belief_decoder_config.pb.h"
#include "mediapipe/modules/objectron/calculators/decoder.h"
#include "mediapipe/modules/objectron/calculators/tensor_util.h"
#include "mediapipe/modules/objectron/calculators/tensors_to_objects_calculator.pb.h"

namespace {
constexpr char kInputStreamTag[] = "TENSORS";
constexpr char kOutputStreamTag[] = "ANNOTATIONS";

// Each detection object will be assigned an unique id that starts from 1.
static int object_id = 0;

inline int GetNextObjectId() { return ++object_id; }
}  // namespace

namespace mediapipe {

// Convert result Tensors from deep pursuit 3d model into FrameAnnotation.
//
// Input:
//  TENSORS - Vector of Tensor of type kFloat32.
// Output:
//  ANNOTATIONS - Result FrameAnnotation.
//
// Usage example:
// node {
//   calculator: "TensorsToObjectsCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "ANNOTATIONS:annotations"
// }
class TensorsToObjectsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          FrameAnnotation* output_objects);
  absl::Status LoadOptions(CalculatorContext* cc);
  // Takes point_3d in FrameAnnotation, projects to 2D, and overwrite the
  // point_2d field with the projection.
  void Project3DTo2D(bool portrait, FrameAnnotation* annotation) const;
  // Increment and assign object ID for each detected object.
  // In a single MediaPipe session, the IDs are unique.
  // Also assign timestamp for the FrameAnnotation to be the input packet
  // timestamp.
  void AssignObjectIdAndTimestamp(int64 timestamp_us,
                                  FrameAnnotation* annotation);

  int num_classes_ = 0;
  int num_keypoints_ = 0;

  ::mediapipe::TensorsToObjectsCalculatorOptions options_;
  std::unique_ptr<Decoder> decoder_;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> projection_matrix_;
};
REGISTER_CALCULATOR(TensorsToObjectsCalculator);

absl::Status TensorsToObjectsCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kInputStreamTag)) {
    cc->Inputs().Tag(kInputStreamTag).Set<std::vector<mediapipe::Tensor>>();
  }

  if (cc->Outputs().HasTag(kOutputStreamTag)) {
    cc->Outputs().Tag(kOutputStreamTag).Set<FrameAnnotation>();
  }
  return absl::OkStatus();
}

absl::Status TensorsToObjectsCalculator::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadOptions(cc));
  // clang-format off
  projection_matrix_ <<
      1.5731,   0,       0, 0,
      0,   2.0975,       0, 0,
      0,        0, -1.0002, -0.2,
      0,        0,      -1, 0;
  // clang-format on
  decoder_ = absl::make_unique<Decoder>(
      BeliefDecoderConfig(options_.decoder_config()));

  return absl::OkStatus();
}

absl::Status TensorsToObjectsCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputStreamTag).IsEmpty()) {
    return absl::OkStatus();
  }

  auto output_objects = absl::make_unique<FrameAnnotation>();

  MP_RETURN_IF_ERROR(ProcessCPU(cc, output_objects.get()));

  // Output
  if (cc->Outputs().HasTag(kOutputStreamTag)) {
    cc->Outputs()
        .Tag(kOutputStreamTag)
        .Add(output_objects.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status TensorsToObjectsCalculator::ProcessCPU(
    CalculatorContext* cc, FrameAnnotation* output_objects) {
  const auto& input_tensors =
      cc->Inputs().Tag(kInputStreamTag).Get<std::vector<mediapipe::Tensor>>();

  cv::Mat prediction_heatmap = ConvertTensorToCvMat(input_tensors[0]);
  cv::Mat offsetmap = ConvertTensorToCvMat(input_tensors[1]);

  *output_objects =
      decoder_->DecodeBoundingBoxKeypoints(prediction_heatmap, offsetmap);
  auto status = decoder_->Lift2DTo3D(projection_matrix_, /*portrait*/ true,
                                     output_objects);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }
  Project3DTo2D(/*portrait*/ true, output_objects);
  AssignObjectIdAndTimestamp(cc->InputTimestamp().Microseconds(),
                             output_objects);

  return absl::OkStatus();
}

absl::Status TensorsToObjectsCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status TensorsToObjectsCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<::mediapipe::TensorsToObjectsCalculatorOptions>();

  num_classes_ = options_.num_classes();
  num_keypoints_ = options_.num_keypoints();

  // Currently only support 2D when num_values_per_keypoint equals to 2.
  CHECK_EQ(options_.num_values_per_keypoint(), 2);

  return absl::OkStatus();
}

void TensorsToObjectsCalculator::Project3DTo2D(
    bool portrait, FrameAnnotation* annotation) const {
  for (auto& ann : *annotation->mutable_annotations()) {
    for (auto& key_point : *ann.mutable_keypoints()) {
      Eigen::Vector4f point3d;
      point3d << key_point.point_3d().x(), key_point.point_3d().y(),
          key_point.point_3d().z(), 1.0f;
      Eigen::Vector4f point3d_projection = projection_matrix_ * point3d;
      float u, v;
      const float inv_w = 1.0f / point3d_projection(3);
      if (portrait) {
        u = (point3d_projection(1) * inv_w + 1.0f) * 0.5f;
        v = (point3d_projection(0) * inv_w + 1.0f) * 0.5f;
      } else {
        u = (point3d_projection(0) * inv_w + 1.0f) * 0.5f;
        v = (1.0f - point3d_projection(1) * inv_w) * 0.5f;
      }
      key_point.mutable_point_2d()->set_x(u);
      key_point.mutable_point_2d()->set_y(v);
    }
  }
}

void TensorsToObjectsCalculator::AssignObjectIdAndTimestamp(
    int64 timestamp_us, FrameAnnotation* annotation) {
  for (auto& ann : *annotation->mutable_annotations()) {
    ann.set_object_id(GetNextObjectId());
  }
  annotation->set_timestamp(timestamp_us);
}

}  // namespace mediapipe
