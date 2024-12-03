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
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_base.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"

// This code should be enabled as soon as TensorFlow version, which mediapipe
// uses, will include this module.
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#endif

namespace tflite {
namespace gpu {
namespace {

// TODO: Find a better place for these utility functions.
void UpdateShapes(const tflite::Interpreter& interpreter,
                  const std::vector<int>& indices,
                  std::vector<std::vector<int>>* shapes) {
  shapes->resize(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    const TfLiteTensor* tensor = interpreter.tensor(indices[i]);
    shapes->at(i).resize(tensor->dims->size);
    for (int j = 0; j < tensor->dims->size; ++j) {
      shapes->at(i)[j] = tensor->dims->data[j];
    }
  }
}

absl::Status InitializeShapes(const tflite::FlatBufferModel& flatbuffer,
                              const tflite::OpResolver& op_resolver,
                              std::vector<std::vector<int>>* input_shapes,
                              std::vector<std::vector<int>>* output_shapes) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("Unable to prepare TfLite interpreter.");
  }
  UpdateShapes(*interpreter, interpreter->inputs(), input_shapes);
  UpdateShapes(*interpreter, interpreter->outputs(), output_shapes);
  return absl::OkStatus();
}

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

#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)

cl::InferenceOptions GetClInferenceOptions(const InferenceOptions& options) {
  cl::InferenceOptions result{};
  result.priority1 = options.priority1;
  result.priority2 = options.priority2;
  result.priority3 = options.priority3;
  result.usage = options.usage;
  return result;
}

absl::Status VerifyShapes(const std::vector<TensorObjectDef>& actual,
                          const std::vector<BHWC>& expected) {
  RET_CHECK_EQ(actual.size(), expected.size());
  const int size = actual.size();
  for (int i = 0; i < size; ++i) {
    const auto& dims = actual[i].dimensions;
    const BHWC& bhwc = expected[i];
    RET_CHECK(dims.b == bhwc.b && dims.h == bhwc.h && dims.w == bhwc.w &&
              dims.c == bhwc.c);
  }
  return absl::OkStatus();
}

#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)

}  // namespace

absl::Status TFLiteGPURunner::InitializeWithModel(
    const tflite::FlatBufferModel& flatbuffer,
    const tflite::OpResolver& op_resolver, bool allow_quant_ops) {
  // GraphFloat32 is created twice because, when OpenCL and OpenGL backends are
  // initialized, different backend-specific graph transformations happen
  // in-place. As GraphFloat32 is not copyable by design, we keep two copies of
  // the graph until inference is built. This decision doesn't affect the amount
  // of run time memory used, because both graph_gl_ and graph_cl_ are deleted
  // in the end of the initialization stage.
  graph_gl_ = std::make_unique<GraphFloat32>();
  graph_cl_ = std::make_unique<GraphFloat32>();
  MP_RETURN_IF_ERROR(BuildFromFlatBuffer(flatbuffer, op_resolver,
                                         graph_gl_.get(), allow_quant_ops));
  MP_RETURN_IF_ERROR(BuildFromFlatBuffer(flatbuffer, op_resolver,
                                         graph_cl_.get(), allow_quant_ops));

  for (const auto& input : graph_gl_->inputs()) {
    input_shapes_.push_back(input->tensor.shape);
  }
  for (const auto& output : graph_gl_->outputs()) {
    output_shapes_.push_back(output->tensor.shape);
  }
  MP_RETURN_IF_ERROR(InitializeShapes(flatbuffer, op_resolver,
                                      &input_shape_from_model_,
                                      &output_shape_from_model_));
  return absl::OkStatus();
}

absl::StatusOr<int64_t> TFLiteGPURunner::GetInputElements(int id) {
  if (id >= input_shapes_.size()) {
    return absl::InternalError("Wrong input tensor id.");
  } else {
    return input_shapes_[id].DimensionsProduct();
  }
}

absl::StatusOr<int64_t> TFLiteGPURunner::GetOutputElements(int id) {
  if (id >= output_shapes_.size()) {
    return absl::InternalError("Wrong output tensor id.");
  } else {
    return output_shapes_[id].DimensionsProduct();
  }
}

absl::Status TFLiteGPURunner::Build() {
  // 1. Prepare inference builder.
  std::unique_ptr<InferenceBuilder> builder;
  // By default, we try CL first & fall back to GL if that fails.
  if (opencl_is_forced_) {
    MP_RETURN_IF_ERROR(InitializeOpenCL(&builder));
    // Only OpenCL delegate supports serializations currently.
    is_cl_used_ = true;
  } else if (opengl_is_forced_) {
    MP_RETURN_IF_ERROR(InitializeOpenGL(&builder));
  } else {
    // try to build OpenCL first. If something goes wrong, fall back to OpenGL.
    absl::Status status = InitializeOpenCL(&builder);
    if (status.ok()) {
      VLOG(2) << "OpenCL backend is used.";
      is_cl_used_ = true;
    } else {
      VLOG(2) << "Falling back to OpenGL: " << status.message();
      MP_RETURN_IF_ERROR(InitializeOpenGL(&builder));
    }
  }

  // GL graph not needed anymore, CL graph maybe needed for serialized model
  // calculation.
  graph_gl_.reset(nullptr);

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

absl::Status TFLiteGPURunner::BindSSBOToInputTensor(GLuint ssbo_id,
                                                    int input_id) {
  OpenGlBuffer buffer;
  buffer.id = ssbo_id;
  return runner_->SetInputObject(input_id, std::move(buffer));
}

absl::Status TFLiteGPURunner::BindSSBOToOutputTensor(GLuint ssbo_id,
                                                     int output_id) {
  OpenGlBuffer buffer;
  buffer.id = ssbo_id;
  return runner_->SetOutputObject(output_id, std::move(buffer));
}

absl::Status TFLiteGPURunner::Invoke() { return runner_->Run(); }

absl::Status TFLiteGPURunner::InitializeOpenGL(
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
  MP_RETURN_IF_ERROR(gl_environment_->NewInferenceBuilder(std::move(*graph_gl_),
                                                          gl_options, builder));
  return absl::OkStatus();
}

absl::Status TFLiteGPURunner::InitializeOpenCL(
    std::unique_ptr<InferenceBuilder>* builder) {
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
  cl::InferenceEnvironmentOptions env_options;
  if (!serialized_binary_cache_.empty()) {
    env_options.serialized_binary_cache = serialized_binary_cache_;
  }
  cl::InferenceEnvironmentProperties properties;
  MP_RETURN_IF_ERROR(
      cl::NewInferenceEnvironment(env_options, &cl_environment_, &properties));

  if (serialized_model_.empty() &&
      opencl_init_from_serialized_model_is_forced_) {
    MP_ASSIGN_OR_RETURN(serialized_model_, GetSerializedModel());
  }

  // Try to initialize from serialized model first.
  if (!serialized_model_.empty()) {
    absl::Status init_status = InitializeOpenCLFromSerializedModel(builder);
    if (init_status.ok()) {
      serialized_model_used_ = true;
      return absl::OkStatus();
    }
    VLOG(2) << "Failed to init from serialized model: [" << init_status
            << "]. Trying to init from scratch.";
  }

  // Initialize from scratch.
  cl::InferenceOptions cl_options = GetClInferenceOptions(options_);
  GraphFloat32 graph_cl;
  MP_RETURN_IF_ERROR(graph_cl_->MakeExactCopy(&graph_cl));
  MP_RETURN_IF_ERROR(cl_environment_->NewInferenceBuilder(
      cl_options, std::move(graph_cl), builder));

  return absl::OkStatus();
#else
  return absl::UnimplementedError("OpenCL is not supported.");
#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
}

absl::Status TFLiteGPURunner::InitializeOpenCLFromSerializedModel(
    std::unique_ptr<InferenceBuilder>* builder) {
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
  RET_CHECK(cl_environment_) << "CL environment is not initialized.";
  MP_RETURN_IF_ERROR(
      cl_environment_->NewInferenceBuilder(serialized_model_, builder));
  MP_RETURN_IF_ERROR(VerifyShapes(builder->get()->inputs(), input_shapes_));
  return VerifyShapes(builder->get()->outputs(), output_shapes_);
#else
  return absl::UnimplementedError(
      "OpenCL and serialized model are not supported.");
#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
}

absl::StatusOr<std::vector<uint8_t>>
TFLiteGPURunner::GetSerializedBinaryCache() {
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
  RET_CHECK(cl_environment_) << "CL environment is not initialized.";
  return cl_environment_->GetSerializedBinaryCache();
#else
  return absl::UnimplementedError("Serialized binary cache is not supported.");
#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
}

void TFLiteGPURunner::SetSerializedBinaryCache(std::vector<uint8_t>&& cache) {
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
  serialized_binary_cache_ = std::move(cache);
#else
  ABSL_LOG(ERROR) << "Serialized binary cache is not supported.";
#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
}

absl::StatusOr<std::vector<uint8_t>> TFLiteGPURunner::GetSerializedModel() {
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
  if (serialized_model_used_) {
    return serialized_model_;
  }
  RET_CHECK(graph_cl_) << "CL graph is not initialized.";
  RET_CHECK(cl_environment_) << "CL environment is not initialized.";

  GraphFloat32 graph_cl;
  MP_RETURN_IF_ERROR(graph_cl_->MakeExactCopy(&graph_cl));
  cl::InferenceOptions cl_options = GetClInferenceOptions(options_);
  std::vector<uint8_t> serialized_model;
  MP_RETURN_IF_ERROR(cl_environment_->BuildSerializedModel(
      cl_options, std::move(graph_cl), &serialized_model));
  return serialized_model;
#else
  return absl::UnimplementedError("Serialized model is not supported.");
#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
}

void TFLiteGPURunner::SetSerializedModel(
    std::vector<uint8_t>&& serialized_model) {
#if defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
  serialized_model_ = std::move(serialized_model);
  serialized_model_used_ = false;
#else
  ABSL_LOG(ERROR) << "Serialized model is not supported.";
#endif  // defined(__ANDROID__) || defined(MEDIAPIPE_CHROMIUMOS)
}

}  // namespace gpu
}  // namespace tflite
