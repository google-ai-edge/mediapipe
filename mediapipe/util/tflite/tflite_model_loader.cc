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

#include "mediapipe/util/tflite/tflite_model_loader.h"

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

absl::StatusOr<api2::Packet<TfLiteModelPtr>> TfLiteModelLoader::LoadFromPath(
    const std::string& path) {
  std::string model_path = path;

  std::string model_blob;
  auto status_or_content =
      mediapipe::GetResourceContents(model_path, &model_blob);
  // TODO: get rid of manual resolving with PathToResourceAsFile
  // as soon as it's incorporated into GetResourceContents.
  if (!status_or_content.ok()) {
    ASSIGN_OR_RETURN(auto resolved_path,
                     mediapipe::PathToResourceAsFile(model_path));
    VLOG(2) << "Loading the model from " << resolved_path;
    MP_RETURN_IF_ERROR(
        mediapipe::GetResourceContents(resolved_path, &model_blob));
  }

  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      model_blob.data(), model_blob.size());
  RET_CHECK(model) << "Failed to load model from path " << model_path;
  return api2::MakePacket<TfLiteModelPtr>(
      model.release(),
      [model_blob = std::move(model_blob)](tflite::FlatBufferModel* model) {
        // It's required that model_blob is deleted only after
        // model is deleted, hence capturing model_blob.
        delete model;
      });
}

}  // namespace mediapipe
