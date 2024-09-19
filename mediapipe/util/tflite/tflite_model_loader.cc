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
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/util/tflite/error_reporter.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/model_builder.h"

namespace mediapipe {

using ::mediapipe::util::tflite::ErrorReporter;
using ::tflite::Allocation;
using ::tflite::FlatBufferModel;
using ::tflite::MMAPAllocation;

absl::StatusOr<api2::Packet<TfLiteModelPtr>> TfLiteModelLoader::LoadFromPath(
    const Resources& resources, const std::string& path, bool try_mmap) {
  std::string model_path = path;

  bool file_exists = file::Exists(model_path).ok();
  if (!file_exists) {
    // TODO: get rid of manual resolving with PathToResourceAsFile
    // as soon as it's incorporated into GetResourceContents.
    absl::StatusOr<std::string> resolved_model_path =
        mediapipe::PathToResourceAsFile(model_path);
    if (resolved_model_path.ok()) {
      VLOG(2) << "Loading the model from " << model_path;
      model_path = *std::move(resolved_model_path);
      file_exists = true;
    }
  }

  // Try to memory map file if available. Falls back to loading from buffer on
  // error.
  if (file_exists && try_mmap && MMAPAllocation::IsSupported()) {
    ErrorReporter error_reporter;
    std::unique_ptr<Allocation> allocation =
        std::make_unique<MMAPAllocation>(model_path.c_str(), &error_reporter);

    if (!error_reporter.HasError()) {
      auto model = FlatBufferModel::BuildFromAllocation(std::move(allocation));
      if (model) {
        return api2::MakePacket<TfLiteModelPtr>(
            model.release(), [](FlatBufferModel* model) { delete model; });
      }
    }

    ABSL_LOG(WARNING) << "Failed to memory map model from path '" << model_path
                      << "'; falling back to loading from buffer. Error: "
                      << error_reporter.message();
  }

  // Load model resource.
  MP_ASSIGN_OR_RETURN(std::unique_ptr<Resource> model_resource,
                      resources.Get(model_path));
  absl::string_view model_view = model_resource->ToStringView();
  auto model = FlatBufferModel::VerifyAndBuildFromBuffer(model_view.data(),
                                                         model_view.size());

  RET_CHECK(model) << "Failed to load model from path " << model_path;
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
