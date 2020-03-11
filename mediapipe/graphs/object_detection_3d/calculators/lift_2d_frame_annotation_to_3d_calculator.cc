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
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/graphs/object_detection_3d/calculators/annotation_data.pb.h"
#include "mediapipe/graphs/object_detection_3d/calculators/decoder.h"
#include "mediapipe/graphs/object_detection_3d/calculators/lift_2d_frame_annotation_to_3d_calculator.pb.h"
#include "mediapipe/graphs/object_detection_3d/calculators/tensor_util.h"

namespace {
constexpr char kInputStreamTag[] = "FRAME_ANNOTATION";
constexpr char kOutputStreamTag[] = "LIFTED_FRAME_ANNOTATION";

// Each detection object will be assigned an unique id that starts from 1.
static int object_id = 0;

inline int GetNextObjectId() { return ++object_id; }
}  // namespace

namespace mediapipe {

// Lifted the 2D points in a tracked frame annotation to 3D.
//
// Input:
//  FRAME_ANNOTATIONS - Frame annotations with detected 2D points
// Output:
//  LIFTED_FRAME_ANNOTATIONS - Result FrameAnnotation with lifted 3D points.
//
// Usage example:
// node {
//   calculator: "Lift2DFrameAnnotationTo3DCalculator"
//   input_stream: "FRAME_ANNOTATIONS:tracked_annotations"
//   output_stream: "LIFTED_FRAME_ANNOTATIONS:lifted_3d_annotations"
// }
class Lift2DFrameAnnotationTo3DCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status ProcessCPU(CalculatorContext* cc,
                                 FrameAnnotation* output_objects);
  ::mediapipe::Status LoadOptions(CalculatorContext* cc);

  // Increment and assign object ID for each detected object.
  // In a single MediaPipe session, the IDs are unique.
  // Also assign timestamp for the FrameAnnotation to be the input packet
  // timestamp.
  void AssignObjectIdAndTimestamp(int64 timestamp_us,
                                  FrameAnnotation* annotation);
  std::unique_ptr<Decoder> decoder_;
  ::mediapipe::Lift2DFrameAnnotationTo3DCalculatorOptions options_;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> projection_matrix_;
};
REGISTER_CALCULATOR(Lift2DFrameAnnotationTo3DCalculator);

::mediapipe::Status Lift2DFrameAnnotationTo3DCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kInputStreamTag));
  RET_CHECK(cc->Outputs().HasTag(kOutputStreamTag));
  cc->Inputs().Tag(kInputStreamTag).Set<FrameAnnotation>();
  cc->Outputs().Tag(kOutputStreamTag).Set<FrameAnnotation>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status Lift2DFrameAnnotationTo3DCalculator::Open(
    CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadOptions(cc));
  // clang-format off
  projection_matrix_ <<
      1.5731,   0,       0,    0,
      0,   2.0975,       0,    0,
      0,        0, -1.0002, -0.2,
      0,        0,      -1,    0;
  // clang-format on

  decoder_ = absl::make_unique<Decoder>(
      BeliefDecoderConfig(options_.decoder_config()));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status Lift2DFrameAnnotationTo3DCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputStreamTag).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }

  auto output_objects = absl::make_unique<FrameAnnotation>();

  MP_RETURN_IF_ERROR(ProcessCPU(cc, output_objects.get()));

  // Output
  if (cc->Outputs().HasTag(kOutputStreamTag)) {
    cc->Outputs()
        .Tag(kOutputStreamTag)
        .Add(output_objects.release(), cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status Lift2DFrameAnnotationTo3DCalculator::ProcessCPU(
    CalculatorContext* cc, FrameAnnotation* output_objects) {
  const auto& input_frame_annotations =
      cc->Inputs().Tag(kInputStreamTag).Get<FrameAnnotation>();
  // Copy the input frame annotation to the output
  *output_objects = input_frame_annotations;

  auto status = decoder_->Lift2DTo3D(projection_matrix_, /*portrait*/ true,
                                     output_objects);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }
  AssignObjectIdAndTimestamp(cc->InputTimestamp().Microseconds(),
                             output_objects);

  return ::mediapipe::OkStatus();
}

::mediapipe::Status Lift2DFrameAnnotationTo3DCalculator::Close(
    CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

::mediapipe::Status Lift2DFrameAnnotationTo3DCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::Lift2DFrameAnnotationTo3DCalculatorOptions>();

  return ::mediapipe::OkStatus();
}

void Lift2DFrameAnnotationTo3DCalculator::AssignObjectIdAndTimestamp(
    int64 timestamp_us, FrameAnnotation* annotation) {
  for (auto& ann : *annotation->mutable_annotations()) {
    ann.set_object_id(GetNextObjectId());
  }
  annotation->set_timestamp(timestamp_us);
}

}  // namespace mediapipe
