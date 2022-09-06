/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/core/model_resources.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/external_file_handler.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/util/tflite/error_reporter.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/shims/cc/model_builder.h"
#include "tensorflow/lite/core/shims/cc/tools/verifier.h"

namespace mediapipe {
namespace tasks {
namespace core {

using ::absl::StatusCode;
using ::mediapipe::api2::MakePacket;
using ::mediapipe::api2::Packet;
using ::mediapipe::api2::PacketAdopting;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

bool ModelResources::Verifier::Verify(const char* data, int length,
                                      tflite::ErrorReporter* reporter) {
  return tflite_shims::Verify(data, length, reporter);
}

ModelResources::ModelResources(const std::string& tag,
                               std::unique_ptr<proto::ExternalFile> model_file,
                               Packet<tflite::OpResolver> op_resolver_packet)
    : tag_(tag),
      model_file_(std::move(model_file)),
      op_resolver_packet_(op_resolver_packet) {}

/* static */
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResources::Create(
    const std::string& tag, std::unique_ptr<proto::ExternalFile> model_file,
    std::unique_ptr<tflite::OpResolver> op_resolver) {
  return Create(tag, std::move(model_file),
                PacketAdopting<tflite::OpResolver>(std::move(op_resolver)));
}

/* static */
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResources::Create(
    const std::string& tag, std::unique_ptr<proto::ExternalFile> model_file,
    Packet<tflite::OpResolver> op_resolver_packet) {
  if (model_file == nullptr) {
    return CreateStatusWithPayload(StatusCode::kInvalidArgument,
                                   "The model file proto cannot be nullptr.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (op_resolver_packet.IsEmpty()) {
    return CreateStatusWithPayload(StatusCode::kInvalidArgument,
                                   "The op resolver packet must be non-empty.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  auto model_resources = absl::WrapUnique(
      new ModelResources(tag, std::move(model_file), op_resolver_packet));
  MP_RETURN_IF_ERROR(model_resources->BuildModelFromExternalFileProto());
  return model_resources;
}

const tflite::Model* ModelResources::GetTfLiteModel() const {
#if !TFLITE_IN_GMSCORE
  return model_packet_.Get()->GetModel();
#else
  return tflite::GetModel(model_file_handler_->GetFileContent().data());
#endif
}

absl::Status ModelResources::BuildModelFromExternalFileProto() {
  if (model_file_->has_file_name()) {
    // If the model file name is a relative path, searches the file in a
    // platform-specific location and returns the absolute path on success.
    ASSIGN_OR_RETURN(std::string path_to_resource,
                     mediapipe::PathToResourceAsFile(model_file_->file_name()));
    model_file_->set_file_name(path_to_resource);
  }
  ASSIGN_OR_RETURN(
      model_file_handler_,
      ExternalFileHandler::CreateFromExternalFile(model_file_.get()));
  const char* buffer_data = model_file_handler_->GetFileContent().data();
  size_t buffer_size = model_file_handler_->GetFileContent().size();
  // Verifies that the supplied buffer refers to a valid flatbuffer model,
  // and that it uses only operators that are supported by the OpResolver
  // that was passed to the ModelResources constructor, and then builds
  // the model from the buffer.
  auto model = tflite_shims::FlatBufferModel::VerifyAndBuildFromBuffer(
      buffer_data, buffer_size, &verifier_, &error_reporter_);
  if (model == nullptr) {
    static constexpr char kInvalidFlatbufferMessage[] =
        "The model is not a valid Flatbuffer";
    // To be replaced with a proper switch-case when TFLite model builder
    // returns a `MediaPipeTasksStatus` code capturing this type of error.
    if (absl::StrContains(error_reporter_.message(),
                          kInvalidFlatbufferMessage)) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument, error_reporter_.message(),
          MediaPipeTasksStatus::kInvalidFlatBufferError);
    } else if (absl::StrContains(error_reporter_.message(),
                                 "Error loading model from buffer")) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument, kInvalidFlatbufferMessage,
          MediaPipeTasksStatus::kInvalidFlatBufferError);
    } else {
      return CreateStatusWithPayload(
          StatusCode::kUnknown,
          absl::StrCat(
              "Could not build model from the provided pre-loaded flatbuffer: ",
              error_reporter_.message()));
    }
  }

  model_packet_ = MakePacket<ModelPtr>(
      model.release(),
      [](tflite_shims::FlatBufferModel* model) { delete model; });
  ASSIGN_OR_RETURN(auto model_metadata_extractor,
                   metadata::ModelMetadataExtractor::CreateFromModelBuffer(
                       buffer_data, buffer_size));
  metadata_extractor_packet_ = PacketAdopting<metadata::ModelMetadataExtractor>(
      std::move(model_metadata_extractor));
  return absl::OkStatus();
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
