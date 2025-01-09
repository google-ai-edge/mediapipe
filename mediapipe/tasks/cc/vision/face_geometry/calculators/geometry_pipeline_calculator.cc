// Copyright 2023 The MediaPipe Authors.
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
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/external_file_handler.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/calculators/geometry_pipeline_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/validation_utils.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.pb.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe::tasks::vision::face_geometry {
namespace {

static constexpr char kEnvironmentTag[] = "ENVIRONMENT";
static constexpr char kImageSizeTag[] = "IMAGE_SIZE";
static constexpr char kMultiFaceGeometryTag[] = "MULTI_FACE_GEOMETRY";
static constexpr char kMultiFaceLandmarksTag[] = "MULTI_FACE_LANDMARKS";
static constexpr char kFaceGeometryTag[] = "FACE_GEOMETRY";
static constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";

using ::mediapipe::tasks::vision::face_geometry::proto::Environment;
using ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;
using ::mediapipe::tasks::vision::face_geometry::proto::
    GeometryPipelineMetadata;

absl::Status SanityCheck(CalculatorContract* cc) {
  if (!(cc->Inputs().HasTag(kFaceLandmarksTag) ^
        cc->Inputs().HasTag(kMultiFaceLandmarksTag))) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Only one of %s and %s can be set at a time.",
                        kFaceLandmarksTag, kMultiFaceLandmarksTag));
  }
  if (!(cc->Outputs().HasTag(kFaceGeometryTag) ^
        cc->Outputs().HasTag(kMultiFaceGeometryTag))) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Only one of %s and %s can be set at a time.",
                        kFaceGeometryTag, kMultiFaceGeometryTag));
  }
  if (cc->Inputs().HasTag(kFaceLandmarksTag) !=
      cc->Outputs().HasTag(kFaceGeometryTag)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "%s and %s must both be set or neither be set and a time.",
            kFaceLandmarksTag, kFaceGeometryTag));
  }
  if (cc->Inputs().HasTag(kMultiFaceLandmarksTag) !=
      cc->Outputs().HasTag(kMultiFaceGeometryTag)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "%s and %s must both be set or neither be set and a time.",
            kMultiFaceLandmarksTag, kMultiFaceGeometryTag));
  }
  return absl::OkStatus();
}

// A calculator that renders a visual effect for multiple faces. Support single
// face landmarks or multiple face landmarks.
//
// Inputs:
//   IMAGE_SIZE (`std::pair<int, int>`, required):
//     The size of the current frame. The first element of the pair is the frame
//     width; the other one is the frame height.
//
//     The face landmarks should have been detected on a frame with the same
//     ratio. If used as-is, the resulting face geometry visualization should be
//     happening on a frame with the same ratio as well.
//
//   MULTI_FACE_LANDMARKS (`std::vector<NormalizedLandmarkList>`, optional):
//     A vector of face landmark lists. If connected, the output stream
//     MULTI_FACE_GEOMETRY must be connected.
//   FACE_LANDMARKS (NormalizedLandmarkList, optional):
//     A NormalizedLandmarkList of single face landmark lists. If connected, the
//     output stream FACE_GEOMETRY must be connected.
//
// Input side packets:
//   ENVIRONMENT (`proto::Environment`, required)
//     Describes an environment; includes the camera frame origin point location
//     as well as virtual camera parameters.
//
// Output:
//   MULTI_FACE_GEOMETRY (`std::vector<FaceGeometry>`, optional):
//     A vector of face geometry data if MULTI_FACE_LANDMARKS is connected .
//   FACE_GEOMETRY (FaceGeometry, optional):
//     A FaceGeometry of the face landmarks if FACE_LANDMARKS is connected.
//
// Options:
//   metadata_file (`ExternalFile`, optional):
//     Defines an ExternalFile for the geometry pipeline metadata file.
//
//     The geometry pipeline metadata file format must be the binary
//     `GeometryPipelineMetadata` proto.
//
class GeometryPipelineCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kEnvironmentTag).Set<Environment>();
    MP_RETURN_IF_ERROR(SanityCheck(cc));
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    if (cc->Inputs().HasTag(kMultiFaceLandmarksTag)) {
      cc->Inputs()
          .Tag(kMultiFaceLandmarksTag)
          .Set<std::vector<mediapipe::NormalizedLandmarkList>>();
      cc->Outputs().Tag(kMultiFaceGeometryTag).Set<std::vector<FaceGeometry>>();
      return absl::OkStatus();
    } else {
      cc->Inputs()
          .Tag(kFaceLandmarksTag)
          .Set<mediapipe::NormalizedLandmarkList>();
      cc->Outputs().Tag(kFaceGeometryTag).Set<FaceGeometry>();
      return absl::OkStatus();
    }
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));

    const auto& options = cc->Options<FaceGeometryPipelineCalculatorOptions>();

    MP_ASSIGN_OR_RETURN(
        GeometryPipelineMetadata metadata,
        ReadMetadataFromFile(options.metadata_file()),
        _ << "Failed to read the geometry pipeline metadata from file!");

    MP_RETURN_IF_ERROR(ValidateGeometryPipelineMetadata(metadata))
        << "Invalid geometry pipeline metadata!";

    const Environment& environment =
        cc->InputSidePackets().Tag(kEnvironmentTag).Get<Environment>();

    MP_RETURN_IF_ERROR(ValidateEnvironment(environment))
        << "Invalid environment!";

    MP_ASSIGN_OR_RETURN(geometry_pipeline_,
                        CreateGeometryPipeline(environment, metadata),
                        _ << "Failed to create a geometry pipeline!");
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Both the `IMAGE_SIZE` and either the `FACE_LANDMARKS` or
    // `MULTI_FACE_LANDMARKS` streams are required to have a non-empty packet.
    // In case this requirement is not met, there's nothing to be processed at
    // the current timestamp and we return early (checked here and below).
    if (cc->Inputs().Tag(kImageSizeTag).IsEmpty()) {
      return absl::OkStatus();
    }

    const auto& image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

    if (cc->Inputs().HasTag(kMultiFaceLandmarksTag)) {
      if (cc->Inputs().Tag(kMultiFaceLandmarksTag).IsEmpty()) {
        return absl::OkStatus();
      }

      const auto& multi_face_landmarks =
          cc->Inputs()
              .Tag(kMultiFaceLandmarksTag)
              .Get<std::vector<mediapipe::NormalizedLandmarkList>>();

      auto multi_face_geometry = absl::make_unique<std::vector<FaceGeometry>>();

      MP_ASSIGN_OR_RETURN(
          *multi_face_geometry,
          geometry_pipeline_->EstimateFaceGeometry(
              multi_face_landmarks,  //
              /*frame_width*/ image_size.first,
              /*frame_height*/ image_size.second),
          _ << "Failed to estimate face geometry for multiple faces!");

      cc->Outputs()
          .Tag(kMultiFaceGeometryTag)
          .AddPacket(mediapipe::Adopt<std::vector<FaceGeometry>>(
                         multi_face_geometry.release())
                         .At(cc->InputTimestamp()));
    } else if (cc->Inputs().HasTag(kFaceLandmarksTag)) {
      if (cc->Inputs().Tag(kFaceLandmarksTag).IsEmpty()) {
        return absl::OkStatus();
      }

      const auto& face_landmarks =
          cc->Inputs()
              .Tag(kFaceLandmarksTag)
              .Get<mediapipe::NormalizedLandmarkList>();

      MP_ASSIGN_OR_RETURN(
          std::vector<FaceGeometry> multi_face_geometry,
          geometry_pipeline_->EstimateFaceGeometry(
              {face_landmarks},  //
              /*frame_width*/ image_size.first,
              /*frame_height*/ image_size.second),
          _ << "Failed to estimate face geometry for multiple faces!");

      cc->Outputs()
          .Tag(kFaceGeometryTag)
          .AddPacket(mediapipe::MakePacket<FaceGeometry>(multi_face_geometry[0])
                         .At(cc->InputTimestamp()));
    }

    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  static absl::StatusOr<GeometryPipelineMetadata> ReadMetadataFromFile(
      const core::proto::ExternalFile& metadata_file) {
    MP_ASSIGN_OR_RETURN(
        const auto file_handler,
        core::ExternalFileHandler::CreateFromExternalFile(&metadata_file));

    GeometryPipelineMetadata metadata;
    RET_CHECK(
        metadata.ParseFromString(std::string(file_handler->GetFileContent())))
        << "Failed to parse a metadata proto from a binary blob!";

    return metadata;
  }

  std::unique_ptr<GeometryPipeline> geometry_pipeline_;
};

}  // namespace

using FaceGeometryPipelineCalculator = GeometryPipelineCalculator;

REGISTER_CALCULATOR(
    ::mediapipe::tasks::vision::face_geometry::FaceGeometryPipelineCalculator);

}  // namespace mediapipe::tasks::vision::face_geometry
