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

#include "mediapipe/util/tflite/tflite_gpu_runner.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/substitute.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace gpu {
namespace {

ObjectDef GetSSBOObjectDef(int channels) {
  ObjectDef gpu_object_def;
  gpu_object_def.data_type = DataType::FLOAT32;
  gpu_object_def.data_layout = DataLayout::BHWC;
  if (channels == 4) {
    gpu_object_def.data_layout = DataLayout::DHWC4;
  }
  gpu_object_def.object_type = ObjectType::OPENGL_SSBO;
  gpu_object_def.user_provided = true;
  return gpu_object_def;
}

}  // namespace

mediapipe::Status TFLiteGPURunner::InitializeWithModel(
    const tflite::FlatBufferModel& flatbuffer,
    const tflite::OpResolver& op_resolver) {
  for (const auto& input : graph_->inputs()) {
    input_shapes_.push_back(input->tensor.shape);
  }
  for (const auto& output : graph_->outputs()) {
    output_shapes_.push_back(output->tensor.shape);
  }
  return absl::OkStatus();
}

mediapipe::StatusOr<int64_t> TFLiteGPURunner::GetInputElements(int id) {
  if (id >= input_shapes_.size()) {
    return ::mediapipe::InternalError("Wrong input tensor id.");
  } else {
    return input_shapes_[id].DimensionsProduct();
  }
}

mediapipe::StatusOr<int64_t> TFLiteGPURunner::GetOutputElements(int id) {
  if (id >= output_shapes_.size()) {
    return ::mediapipe::InternalError("Wrong output tensor id.");
  } else {
    return output_shapes_[id].DimensionsProduct();
  }
}

mediapipe::Status TFLiteGPURunner::Build() {
  // 1. Prepare inference builder.
  std::unique_ptr<InferenceBuilder> builder;
  MP_RETURN_IF_ERROR(InitializeOpenGL(&builder));

  // 2. Describe output/input objects for created builder.
  for (int flow_index = 0; flow_index < input_shapes_.size(); ++flow_index) {
    MP_RETURN_IF_ERROR(builder->SetInputObjectDef(
        flow_index, GetSSBOObjectDef(input_shapes_[flow_index].c)));
  }
  for (int flow_index = 0; flow_index < output_shapes_.size(); ++flow_index) {
    MP_RETURN_IF_ERROR(builder->SetOutputObjectDef(
        flow_index, GetSSBOObjectDef(output_shapes_[flow_index].c)));
  }

  // 3. Build inference runner with the created builder.
  return builder->Build(&runner_);
}

mediapipe::Status TFLiteGPURunner::BindSSBOToInputTensor(GLuint ssbo_id,
                                                         int input_id) {
  OpenGlBuffer buffer;
  buffer.id = ssbo_id;
  return runner_->SetInputObject(input_id, std::move(buffer));
}

mediapipe::Status TFLiteGPURunner::BindSSBOToOutputTensor(GLuint ssbo_id,
                                                          int output_id) {
  OpenGlBuffer buffer;
  buffer.id = ssbo_id;
  return runner_->SetOutputObject(output_id, std::move(buffer));
}

mediapipe::Status TFLiteGPURunner::Invoke() { return runner_->Run(); }

mediapipe::Status TFLiteGPURunner::InitializeOpenGL(
    std::unique_ptr<InferenceBuilder>* builder) {
  gl::InferenceEnvironmentOptions env_options;
  gl::InferenceEnvironmentProperties properties;
  gl::InferenceOptions gl_options;
  gl_options.priority1 = options_.priority1;
  gl_options.priority2 = options_.priority2;
  gl_options.priority3 = options_.priority3;
  gl_options.usage = options_.usage;
  MP_RETURN_IF_ERROR(
      NewInferenceEnvironment(env_options, &gl_environment_, &properties));
  MP_RETURN_IF_ERROR(gl_environment_->NewInferenceBuilder(std::move(*graph_),
                                                          gl_options, builder));
  graph_.release();
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
