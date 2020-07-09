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

#ifndef MEDIAPIPE_CALCULATORS_TFLITE_TFLITE_GPU_RUNNER_H_
#define MEDIAPIPE_CALCULATORS_TFLITE_TFLITE_GPU_RUNNER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#include "tensorflow/lite/model.h"

#ifdef __ANDROID__
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#endif

namespace tflite {
namespace gpu {

// Executes GPU based inference using the TFLite GPU delegate api2.
// Currently supports only GPU inputs/outputs.
//
// Typical order of execution:
// 1. Initialize with the flatbuffer model using InitializeWithModel().
// 2. Bind OpenGL SSBO objects as inputs and outputs using
// BindSSBOToInputTensor() and BindSSBOToOutputTensor().
// 3. Build the inference runner with Build() method.
// 4. Invoke() executes the inference, where inputs and outputs are those which
// were specified earlier. Invoke() may be called in the loop.
//
// Note: All of these need to happen inside MediaPipe's RunInGlContext to make
// sure that all steps from inference construction to execution are made using
// same OpenGL context.
class TFLiteGPURunner {
 public:
  explicit TFLiteGPURunner(const InferenceOptions& options)
      : options_(options) {}

  mediapipe::Status InitializeWithModel(
      const tflite::FlatBufferModel& flatbuffer,
      const tflite::OpResolver& op_resolver);
  mediapipe::Status BindSSBOToInputTensor(GLuint ssbo_id, int input_id);
  mediapipe::Status BindSSBOToOutputTensor(GLuint ssbo_id, int output_id);

  int inputs_size() const { return input_shapes_.size(); }
  int outputs_size() const { return output_shapes_.size(); }

  mediapipe::StatusOr<int64_t> GetInputElements(int id);
  mediapipe::StatusOr<int64_t> GetOutputElements(int id);

  mediapipe::Status Build();
  mediapipe::Status Invoke();

  std::vector<BHWC> GetInputShapes() { return input_shapes_; }
  std::vector<BHWC> GetOutputShapes() { return output_shapes_; }

 private:
  mediapipe::Status InitializeOpenGL(
      std::unique_ptr<InferenceBuilder>* builder);
  mediapipe::Status InitializeOpenCL(
      std::unique_ptr<InferenceBuilder>* builder);

  InferenceOptions options_;
  std::unique_ptr<gl::InferenceEnvironment> gl_environment_;

#ifdef __ANDROID__
  std::unique_ptr<cl::InferenceEnvironment> cl_environment_;
#endif

  // graph_ is maintained temporarily and becomes invalid after runner_ is ready
  std::unique_ptr<GraphFloat32> graph_gl_;
  std::unique_ptr<GraphFloat32> graph_cl_;
  std::unique_ptr<InferenceRunner> runner_;

  // We keep information about input/output shapes, because they are needed
  // after graph_ becomes "converted" into runner_.
  std::vector<BHWC> input_shapes_;
  std::vector<BHWC> output_shapes_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // MEDIAPIPE_CALCULATORS_TFLITE_TFLITE_GPU_RUNNER_H_
