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

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "tensorflow/lite/model_builder.h"

namespace mediapipe {

using ::tflite::FlatBufferModel;

absl::StatusOr<api2::Packet<TfLiteModelPtr>> TfLiteModelLoader::LoadFromPath(
    const Resources& resources, const std::string& path, bool try_mmap) {
  std::string model_path = path;

  // Load model resource.
  MP_ASSIGN_OR_RETURN(
      std::unique_ptr<Resource> model_resource,
      resources.Get(
          model_path,
          Resources::Options{/* read_as_binary= */ true,
                             /* mmap_mode= */ try_mmap
                                 ? std::make_optional(MMapMode::kMMapOrRead)
                                 : std::nullopt}));
  absl::string_view model_view = model_resource->ToStringView();
  auto model = FlatBufferModel::VerifyAndBuildFromBuffer(model_view.data(),
                                                         model_view.size());

  RET_CHECK(model) << "Failed to load model from path (resource ID) "
                   << model_path;
  return api2::MakePacket<TfLiteModelPtr>(
      model.release(), [model_resource = model_resource.release()](
                           FlatBufferModel* model) mutable {
        // It's required that model resource, used for model creation, outlives
        // the created model.
        delete model;
        delete model_resource;
      });
}
}  // namespace mediapipe
