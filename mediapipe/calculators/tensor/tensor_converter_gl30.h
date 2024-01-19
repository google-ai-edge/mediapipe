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
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSOR_CONVERTER_GL_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSOR_CONVERTER_GL_H_

#include <memory>

#include "mediapipe/calculators/tensor/tensor_converter_gpu.h"
#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#include "mediapipe/gpu/gl_calculator_helper.h"

namespace mediapipe {

// Instantiates an OpenGL 3.0-enabled TensorConverterGpu instance.
std::unique_ptr<TensorConverterGpu> CreateTensorConverterGl30(
    GlCalculatorHelper& gpu_helper);

}  // namespace mediapipe
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSOR_CONVERTER_GL_H_
