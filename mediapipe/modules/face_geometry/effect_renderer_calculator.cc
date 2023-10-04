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
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"       // NOTYPO
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"  // NOTYPO
#include "mediapipe/framework/port/opencv_imgproc_inc.h"    // NOTYPO
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/modules/face_geometry/effect_renderer_calculator.pb.h"
#include "mediapipe/modules/face_geometry/libs/effect_renderer.h"
#include "mediapipe/modules/face_geometry/libs/validation_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {
namespace {

static constexpr char kEnvironmentTag[] = "ENVIRONMENT";
static constexpr char kImageGpuTag[] = "IMAGE_GPU";
static constexpr char kMultiFaceGeometryTag[] = "MULTI_FACE_GEOMETRY";

// A calculator that renders a visual effect for multiple faces.
//
// Inputs:
//   IMAGE_GPU (`GpuBuffer`, required):
//     A buffer containing input image.
//
//   MULTI_FACE_GEOMETRY (`std::vector<face_geometry::FaceGeometry>`, optional):
//     A vector of face geometry data.
//
//     If absent, the input GPU buffer is copied over into the output GPU buffer
//     without any effect being rendered.
//
// Input side packets:
//   ENVIRONMENT (`face_geometry::Environment`, required)
//     Describes an environment; includes the camera frame origin point location
//     as well as virtual camera parameters.
//
// Output:
//   IMAGE_GPU (`GpuBuffer`, required):
//     A buffer with a visual effect being rendered for multiple faces.
//
// Options:
//   effect_texture_path (`string`, required):
//     Defines a path for the visual effect texture file. The effect texture is
//     later rendered on top of the effect mesh.
//
//     The texture file format must be supported by the OpenCV image decoder. It
//     must also define either an RGB or an RGBA texture.
//
//   effect_mesh_3d_path (`string`, optional):
//     Defines a path for the visual effect mesh 3D file. The effect mesh is
//     later "attached" to the face and is driven by the face pose
//     transformation matrix.
//
//     The mesh 3D file format must be the binary `face_geometry.Mesh3d` proto.
//
//     If is not present, the runtime face mesh will be used as the effect mesh
//     - this mode is handy for facepaint effects.
//
class EffectRendererCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc))
        << "Failed to update contract for the GPU helper!";

    cc->InputSidePackets()
        .Tag(kEnvironmentTag)
        .Set<face_geometry::Environment>();
    cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
    cc->Inputs()
        .Tag(kMultiFaceGeometryTag)
        .Set<std::vector<face_geometry::FaceGeometry>>();
    cc->Outputs().Tag(kImageGpuTag).Set<GpuBuffer>();

    return mediapipe::GlCalculatorHelper::UpdateContract(cc);
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));

    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc))
        << "Failed to open the GPU helper!";
    return gpu_helper_.RunInGlContext([&]() -> absl::Status {
      const auto& options =
          cc->Options<FaceGeometryEffectRendererCalculatorOptions>();

      const auto& environment = cc->InputSidePackets()
                                    .Tag(kEnvironmentTag)
                                    .Get<face_geometry::Environment>();

      MP_RETURN_IF_ERROR(face_geometry::ValidateEnvironment(environment))
          << "Invalid environment!";

      absl::optional<face_geometry::Mesh3d> effect_mesh_3d;
      if (options.has_effect_mesh_3d_path()) {
        MP_ASSIGN_OR_RETURN(
            effect_mesh_3d, ReadMesh3dFromFile(options.effect_mesh_3d_path()),
            _ << "Failed to read the effect 3D mesh from file!");

        MP_RETURN_IF_ERROR(face_geometry::ValidateMesh3d(*effect_mesh_3d))
            << "Invalid effect 3D mesh!";
      }

      MP_ASSIGN_OR_RETURN(ImageFrame effect_texture,
                          ReadTextureFromFile(options.effect_texture_path()),
                          _ << "Failed to read the effect texture from file!");

      MP_ASSIGN_OR_RETURN(effect_renderer_,
                          CreateEffectRenderer(environment, effect_mesh_3d,
                                               std::move(effect_texture)),
                          _ << "Failed to create the effect renderer!");

      return absl::OkStatus();
    });
  }

  absl::Status Process(CalculatorContext* cc) override {
    // The `IMAGE_GPU` stream is required to have a non-empty packet. In case
    // this requirement is not met, there's nothing to be processed at the
    // current timestamp.
    if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
      return absl::OkStatus();
    }

    return gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      const auto& input_gpu_buffer =
          cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();

      GlTexture input_gl_texture =
          gpu_helper_.CreateSourceTexture(input_gpu_buffer);

      GlTexture output_gl_texture = gpu_helper_.CreateDestinationTexture(
          input_gl_texture.width(), input_gl_texture.height());

      std::vector<face_geometry::FaceGeometry> empty_multi_face_geometry;
      const auto& multi_face_geometry =
          cc->Inputs().Tag(kMultiFaceGeometryTag).IsEmpty()
              ? empty_multi_face_geometry
              : cc->Inputs()
                    .Tag(kMultiFaceGeometryTag)
                    .Get<std::vector<face_geometry::FaceGeometry>>();

      // Validate input multi face geometry data.
      for (const face_geometry::FaceGeometry& face_geometry :
           multi_face_geometry) {
        MP_RETURN_IF_ERROR(face_geometry::ValidateFaceGeometry(face_geometry))
            << "Invalid face geometry!";
      }

      MP_RETURN_IF_ERROR(effect_renderer_->RenderEffect(
          multi_face_geometry, input_gl_texture.width(),
          input_gl_texture.height(), input_gl_texture.target(),
          input_gl_texture.name(), output_gl_texture.target(),
          output_gl_texture.name()))
          << "Failed to render the effect!";

      std::unique_ptr<GpuBuffer> output_gpu_buffer =
          output_gl_texture.GetFrame<GpuBuffer>();

      cc->Outputs()
          .Tag(kImageGpuTag)
          .AddPacket(mediapipe::Adopt<GpuBuffer>(output_gpu_buffer.release())
                         .At(cc->InputTimestamp()));

      output_gl_texture.Release();
      input_gl_texture.Release();

      return absl::OkStatus();
    });
  }

  ~EffectRendererCalculator() {
    gpu_helper_.RunInGlContext([this]() { effect_renderer_.reset(); });
  }

 private:
  static absl::StatusOr<ImageFrame> ReadTextureFromFile(
      const std::string& texture_path) {
    MP_ASSIGN_OR_RETURN(std::string texture_blob,
                        ReadContentBlobFromFile(texture_path),
                        _ << "Failed to read texture blob from file!");

    // Use OpenCV image decoding functionality to finish reading the texture.
    std::vector<char> texture_blob_vector(texture_blob.begin(),
                                          texture_blob.end());
    cv::Mat decoded_mat =
        cv::imdecode(texture_blob_vector, cv::IMREAD_UNCHANGED);

    RET_CHECK(decoded_mat.type() == CV_8UC3 || decoded_mat.type() == CV_8UC4)
        << "Texture must have `char` as the underlying type and "
           "must have either 3 or 4 channels!";

    ImageFormat::Format image_format = ImageFormat::UNKNOWN;
    cv::Mat output_mat;
    switch (decoded_mat.channels()) {
      case 3:
        image_format = ImageFormat::SRGB;
        cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGB);
        break;

      case 4:
        image_format = ImageFormat::SRGBA;
        cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGRA2RGBA);
        break;

      default:
        RET_CHECK_FAIL()
            << "Unexpected number of channels; expected 3 or 4, got "
            << decoded_mat.channels() << "!";
    }

    ImageFrame output_image_frame(image_format, output_mat.size().width,
                                  output_mat.size().height,
                                  ImageFrame::kGlDefaultAlignmentBoundary);

    output_mat.copyTo(formats::MatView(&output_image_frame));

    return output_image_frame;
  }

  static absl::StatusOr<face_geometry::Mesh3d> ReadMesh3dFromFile(
      const std::string& mesh_3d_path) {
    MP_ASSIGN_OR_RETURN(std::string mesh_3d_blob,
                        ReadContentBlobFromFile(mesh_3d_path),
                        _ << "Failed to read mesh 3D blob from file!");

    face_geometry::Mesh3d mesh_3d;
    RET_CHECK(mesh_3d.ParseFromString(mesh_3d_blob))
        << "Failed to parse a mesh 3D proto from a binary blob!";

    return mesh_3d;
  }

  static absl::StatusOr<std::string> ReadContentBlobFromFile(
      const std::string& unresolved_path) {
    MP_ASSIGN_OR_RETURN(
        std::string resolved_path,
        mediapipe::PathToResourceAsFile(unresolved_path),
        _ << "Failed to resolve path! Path = " << unresolved_path);

    std::string content_blob;
    MP_RETURN_IF_ERROR(
        mediapipe::GetResourceContents(resolved_path, &content_blob))
        << "Failed to read content blob! Resolved path = " << resolved_path;

    return content_blob;
  }

  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<face_geometry::EffectRenderer> effect_renderer_;
};

}  // namespace

using FaceGeometryEffectRendererCalculator = EffectRendererCalculator;

REGISTER_CALCULATOR(FaceGeometryEffectRendererCalculator);

}  // namespace mediapipe
