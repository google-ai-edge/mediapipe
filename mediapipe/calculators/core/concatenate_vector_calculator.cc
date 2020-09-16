// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/calculators/core/concatenate_vector_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/integral_types.h"
#include "tensorflow/lite/interpreter.h"

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#endif  //  !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

// Example config:
// node {
//   calculator: "ConcatenateFloatVectorCalculator"
//   input_stream: "float_vector_1"
//   input_stream: "float_vector_2"
//   output_stream: "concatenated_float_vector"
// }
typedef ConcatenateVectorCalculator<float> ConcatenateFloatVectorCalculator;
REGISTER_CALCULATOR(ConcatenateFloatVectorCalculator);

// Example config:
// node {
//   calculator: "ConcatenateInt32VectorCalculator"
//   input_stream: "int32_vector_1"
//   input_stream: "int32_vector_2"
//   output_stream: "concatenated_int32_vector"
// }
typedef ConcatenateVectorCalculator<int32> ConcatenateInt32VectorCalculator;
REGISTER_CALCULATOR(ConcatenateInt32VectorCalculator);

typedef ConcatenateVectorCalculator<uint64> ConcatenateUInt64VectorCalculator;
REGISTER_CALCULATOR(ConcatenateUInt64VectorCalculator);

typedef ConcatenateVectorCalculator<bool> ConcatenateBoolVectorCalculator;
REGISTER_CALCULATOR(ConcatenateBoolVectorCalculator);

// Example config:
// node {
//   calculator: "ConcatenateTfLiteTensorVectorCalculator"
//   input_stream: "tflitetensor_vector_1"
//   input_stream: "tflitetensor_vector_2"
//   output_stream: "concatenated_tflitetensor_vector"
// }
typedef ConcatenateVectorCalculator<TfLiteTensor>
    ConcatenateTfLiteTensorVectorCalculator;
REGISTER_CALCULATOR(ConcatenateTfLiteTensorVectorCalculator);

typedef ConcatenateVectorCalculator<::mediapipe::NormalizedLandmark>
    ConcatenateLandmarkVectorCalculator;
REGISTER_CALCULATOR(ConcatenateLandmarkVectorCalculator);

typedef ConcatenateVectorCalculator<::mediapipe::NormalizedLandmarkList>
    ConcatenateLandmarListVectorCalculator;
REGISTER_CALCULATOR(ConcatenateLandmarListVectorCalculator);

typedef ConcatenateVectorCalculator<mediapipe::ClassificationList>
    ConcatenateClassificationListVectorCalculator;
REGISTER_CALCULATOR(ConcatenateClassificationListVectorCalculator);

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
typedef ConcatenateVectorCalculator<::tflite::gpu::gl::GlBuffer>
    ConcatenateGlBufferVectorCalculator;
REGISTER_CALCULATOR(ConcatenateGlBufferVectorCalculator);
#endif

}  // namespace mediapipe
