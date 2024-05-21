/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/vision/face_stylizer/face_stylizer.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/vision/face_stylizer/face_stylizer.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::face_stylizer {

namespace {

using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::face_stylizer::FaceStylizer;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

FaceStylizer* CppFaceStylizerCreate(const FaceStylizerOptions& options,
                                    char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::face_stylizer::FaceStylizerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);

  auto stylizer = FaceStylizer::Create(std::move(cpp_options));
  if (!stylizer.ok()) {
    ABSL_LOG(ERROR) << "Failed to create FaceStylizer: " << stylizer.status();
    CppProcessError(stylizer.status(), error_msg);
    return nullptr;
  }
  return stylizer->release();
}

int CppFaceStylizerStylize(void* stylizer, const MpImage* image,
                           MpImage* result, char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    ABSL_LOG(ERROR) << "Stylization failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_stylizer = static_cast<FaceStylizer*>(stylizer);
  auto cpp_stylized_image = cpp_stylizer->Stylize(*img);
  if (!cpp_stylized_image.ok()) {
    ABSL_LOG(ERROR) << "Stylization failed: " << cpp_stylized_image.status();
    return CppProcessError(cpp_stylized_image.status(), error_msg);
  }

  if (cpp_stylized_image.value().has_value()) {
    const auto& cpp_stylized_image_frame =
        cpp_stylized_image.value().value().GetImageFrameSharedPtr();

    const int pixel_data_size =
        cpp_stylized_image_frame->PixelDataSizeStoredContiguously();
    auto* pixel_data = new uint8_t[pixel_data_size];
    cpp_stylized_image_frame->CopyToBuffer(pixel_data, pixel_data_size);
    result->image_frame = {.format = static_cast<::ImageFormat>(
                               cpp_stylized_image_frame->Format()),
                           .image_buffer = pixel_data,
                           .width = cpp_stylized_image_frame->Width(),
                           .height = cpp_stylized_image_frame->Height()};
  }

  return 0;
}

void CppFaceStylizerCloseResult(MpImage* result) {
  delete[] result->image_frame.image_buffer;
}

int CppFaceStylizerClose(void* stylizer, char** error_msg) {
  auto cpp_stylizer = static_cast<FaceStylizer*>(stylizer);
  auto result = cpp_stylizer->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close FaceStylizer: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_stylizer;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::face_stylizer

extern "C" {

void* face_stylizer_create(struct FaceStylizerOptions* options,
                           char** error_msg) {
  return mediapipe::tasks::c::vision::face_stylizer::CppFaceStylizerCreate(
      *options, error_msg);
}

int face_stylizer_stylize_image(void* stylizer, const MpImage* image,
                                MpImage* result, char** error_msg) {
  return mediapipe::tasks::c::vision::face_stylizer::CppFaceStylizerStylize(
      stylizer, image, result, error_msg);
}

void face_stylizer_close_result(MpImage* result) {
  mediapipe::tasks::c::vision::face_stylizer::CppFaceStylizerCloseResult(
      result);
}

int face_stylizer_close(void* stylizer, char** error_ms) {
  return mediapipe::tasks::c::vision::face_stylizer::CppFaceStylizerClose(
      stylizer, error_ms);
}

}  // extern "C"
