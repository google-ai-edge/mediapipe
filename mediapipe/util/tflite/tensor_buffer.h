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

#ifndef MEDIAPIPE_UTIL_TFLITE_TENSOR_BUFFER_H_
#define MEDIAPIPE_UTIL_TFLITE_TENSOR_BUFFER_H_

#include "absl/memory/memory.h"
#include "mediapipe/framework/port.h"
#include "tensorflow/lite/interpreter.h"

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
#include "mediapipe/gpu/gl_context.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#endif  //  MEDIAPIPE_DISABLE_GL_COMPUTE

#if defined(MEDIAPIPE_IOS)
#import <Metal/Metal.h>
#endif  // MEDIAPIPE_IOS

namespace mediapipe {

class TensorBuffer {
 public:
  TensorBuffer();
  ~TensorBuffer();

  TensorBuffer(TfLiteTensor& tensor);
  TfLiteTensor* GetTfLiteTensor() { return &cpu_; }
  const TfLiteTensor* GetTfLiteTensor() const { return &cpu_; }

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  TensorBuffer(std::shared_ptr<tflite::gpu::gl::GlBuffer> tensor);
  std::shared_ptr<tflite::gpu::gl::GlBuffer> GetGlBuffer() { return gpu_; }
  const std::shared_ptr<tflite::gpu::gl::GlBuffer> GetGlBuffer() const {
    return gpu_;
  }
  // Example use:
  // auto tensor_buf = TensorBuffer(TensorBuffer::CreateGlBuffer(gl_context));
  static std::shared_ptr<tflite::gpu::gl::GlBuffer> CreateGlBuffer(
      std::shared_ptr<mediapipe::GlContext> context);
#endif  // MEDIAPIPE_DISABLE_GL_COMPUTE

#if defined(MEDIAPIPE_IOS)
  TensorBuffer(id<MTLBuffer> tensor);
  id<MTLBuffer> GetMetalBuffer() { return gpu_; }
  const id<MTLBuffer> GetMetalBuffer() const { return gpu_; }
#endif  // MEDIAPIPE_IOS

  bool UsesGpu() const { return uses_gpu_; }

 private:
  TfLiteTensor cpu_;

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  std::shared_ptr<tflite::gpu::gl::GlBuffer> gpu_;
#endif  // MEDIAPIPE_DISABLE_GL_COMPUTE

#if defined(MEDIAPIPE_IOS)
  typedef id<MTLBuffer> gpu_;
#endif  // MEDIAPIPE_IOS

  bool uses_gpu_ = false;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_TFLITE_TENSOR_BUFFER_H_
