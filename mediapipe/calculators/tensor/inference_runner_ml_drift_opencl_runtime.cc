// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/inference_runner_ml_drift_opencl_runtime.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"  // IWYU pragma: keep
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/model_builder.h"
#include "third_party/GL/gl/include/GLES2/gl2.h"
#include "third_party/ml_drift/cl/api.h"
#include "third_party/ml_drift/common/data_type.h"
#include "third_party/ml_drift/common/model.h"
#include "third_party/ml_drift/common/model_builder.h"
#include "third_party/ml_drift/common/shape.h"
#include "third_party/ml_drift/contrib/tflite_op_resolver.h"
#include "third_party/ml_drift/delegate/api_cl_gl_vk.h"
#include "third_party/ml_drift/delegate/api_common.h"

namespace mediapipe::api2 {

namespace {

using ::ml_drift::BHWC;
using ::ml_drift::DataLayout;
using ::ml_drift::DataType;
using ::ml_drift::GraphFloat32;
using ::ml_drift::InferenceBuilder;
using ::ml_drift::ObjectDef;
using ::ml_drift::ObjectType;
using ::ml_drift::OpenGlBuffer;
using ::ml_drift::cl::InferenceEnvironmentOptions;
using ::ml_drift::cl::InferenceEnvironmentProperties;

ObjectDef GetSSboObjectDef(int channels) {
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

absl::Status InferenceRunnerMlDriftOpenClRuntime::Init(
    const InferenceCalculatorOptions& options,
    Packet<TfLiteModelPtr> model_packet,
    Packet<ml_drift::contrib::TfLiteOpResolver> op_resolver_packet) {
  RET_CHECK_EQ(options.delegate().gpu().api(),
               InferenceCalculatorOptions::Delegate::Gpu::ML_DRIFT_OPENCL);

  bool allow_precision_loss = options.delegate().gpu().allow_precision_loss();

  ml_drift::cl::InferenceOptions mldrift_options;
  mldrift_options.priority1 = allow_precision_loss
                                  ? ml_drift::InferencePriority::MIN_LATENCY
                                  : ml_drift::InferencePriority::MAX_PRECISION;
  mldrift_options.priority2 = ml_drift::InferencePriority::AUTO;
  mldrift_options.priority3 = ml_drift::InferencePriority::AUTO;
  switch (options.delegate().gpu().usage()) {
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::
        FAST_SINGLE_ANSWER: {
      mldrift_options.usage = ml_drift::InferenceUsage::FAST_SINGLE_ANSWER;
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::
        SUSTAINED_SPEED: {
      mldrift_options.usage = ml_drift::InferenceUsage::SUSTAINED_SPEED;
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::UNSPECIFIED: {
      return absl::InternalError("inference usage need to be specified.");
    }
  }

  MP_ASSIGN_OR_RETURN(input_output_tensor_names_,
                      InferenceIoMapper::GetInputOutputTensorNamesFromModel(
                          *model_packet.Get(), op_resolver_packet.Get()));

  MP_ASSIGN_OR_RETURN(
      GraphFloat32 graph_cl,
      InitModelFromFlatBuffer(*model_packet.Get(), op_resolver_packet.Get(),
                              /*allow_quant_ops=*/true));

  tensor_output_shapes_.reserve(output_shapes_.size());
  for (int i = 0; i < output_shapes_.size(); ++i) {
    tensor_output_shapes_.push_back({output_shapes_[i].b, output_shapes_[i].h,
                                     output_shapes_[i].w, output_shapes_[i].c});
  }

  return InitializeMlDriftRuntime(std::move(graph_cl), mldrift_options);
}

// Tensor::GetOpenGlBufferReadView is only defined for OpenGl ES 3.1.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
absl::StatusOr<std::vector<Tensor>> InferenceRunnerMlDriftOpenClRuntime::Run(
    CalculatorContext* cc, const TensorSpan& input_tensors) {
  std::vector<Tensor> output_tensors;
  for (int i = 0; i < input_tensors.size(); ++i) {
    MP_RETURN_IF_ERROR(BindSsboToInputTensor(
        input_tensors[i].GetOpenGlBufferReadView().name(), i));
  }
  output_tensors.reserve(tensor_output_shapes_.size());
  for (int i = 0; i < tensor_output_shapes_.size(); ++i) {
    output_tensors.emplace_back(Tensor::ElementType::kFloat32,
                                tensor_output_shapes_[i]);
    MP_RETURN_IF_ERROR(BindSsboToOutputTensor(
        output_tensors.back().GetOpenGlBufferWriteView().name(), i));
  }
  // Run inference.
  MP_RETURN_IF_ERROR(runner_->Run());
  return output_tensors;
}
#endif

const InputOutputTensorNames&
InferenceRunnerMlDriftOpenClRuntime::GetInputOutputTensorNames() const {
  return input_output_tensor_names_;
}

absl::StatusOr<GraphFloat32>
InferenceRunnerMlDriftOpenClRuntime::InitModelFromFlatBuffer(
    const tflite::FlatBufferModel& flatbuffer,
    const tflite::OpResolver& op_resolver, bool allow_quant_ops) {
  GraphFloat32 graph_cl;
  MP_RETURN_IF_ERROR(
      BuildFromFlatBuffer(flatbuffer, op_resolver, &graph_cl, allow_quant_ops));

  for (const auto& input : graph_cl.inputs()) {
    input_shapes_.push_back(input->tensor.shape);
  }
  for (const auto& output : graph_cl.outputs()) {
    output_shapes_.push_back(output->tensor.shape);
  }
  return graph_cl;
}

absl::Status InferenceRunnerMlDriftOpenClRuntime::InitializeMlDriftRuntime(
    GraphFloat32&& graph_cl, const ml_drift::cl::InferenceOptions& options) {
  // 1. Prepare inference builder.
  std::unique_ptr<InferenceBuilder> builder;

  InferenceEnvironmentOptions env_options;
  InferenceEnvironmentProperties properties;
  MP_RETURN_IF_ERROR(
      NewInferenceEnvironment(env_options, &cl_environment_, &properties));

  // Initialize from scratch.
  MP_RETURN_IF_ERROR(cl_environment_->NewInferenceBuilder(
      options, std::move(graph_cl), &builder));

  // 2. Describe output/input objects for created builder.
  for (int flow_index = 0; flow_index < input_shapes_.size(); ++flow_index) {
    MP_RETURN_IF_ERROR(builder->SetInputObjectDef(
        flow_index, GetSSboObjectDef(input_shapes_[flow_index].c)));
  }
  for (int flow_index = 0; flow_index < output_shapes_.size(); ++flow_index) {
    MP_RETURN_IF_ERROR(builder->SetOutputObjectDef(
        flow_index, GetSSboObjectDef(output_shapes_[flow_index].c)));
  }
  // 3. Build inference runner with the created builder.
  MP_ASSIGN_OR_RETURN(runner_, builder->Build());
  return absl::OkStatus();
}

absl::Status InferenceRunnerMlDriftOpenClRuntime::BindSsboToInputTensor(
    GLuint ssbo_id, int input_id) {
  OpenGlBuffer buffer;
  buffer.id = ssbo_id;
  return runner_->SetInputObject(input_id, std::move(buffer));
}

absl::Status InferenceRunnerMlDriftOpenClRuntime::BindSsboToOutputTensor(
    GLuint ssbo_id, int output_id) {
  OpenGlBuffer buffer;
  buffer.id = ssbo_id;
  return runner_->SetOutputObject(output_id, std::move(buffer));
}

}  // namespace mediapipe::api2
