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

#include <cstdint>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/util/render_data.pb.h"
#include "tensorflow/lite/interpreter.h"

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

namespace mediapipe {

// Example config:
// node {
//   calculator: "ConcatenateFloatVectorCalculator"
//   input_stream: "float_vector_1"
//   input_stream: "float_vector_2"
//   output_stream: "concatenated_float_vector"
// }
typedef ConcatenateVectorCalculator<float> ConcatenateFloatVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateFloatVectorCalculator);

// Example config:
// node {
//   calculator: "ConcatenateInt32VectorCalculator"
//   input_stream: "int32_vector_1"
//   input_stream: "int32_vector_2"
//   output_stream: "concatenated_int32_vector"
// }
typedef ConcatenateVectorCalculator<int32_t> ConcatenateInt32VectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateInt32VectorCalculator);

typedef ConcatenateVectorCalculator<uint64_t> ConcatenateUInt64VectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateUInt64VectorCalculator);

typedef ConcatenateVectorCalculator<bool> ConcatenateBoolVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateBoolVectorCalculator);

typedef ConcatenateVectorCalculator<std::string>
    ConcatenateStringVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateStringVectorCalculator);

// Example config:
// node {
//   calculator: "ConcatenateTfLiteTensorVectorCalculator"
//   input_stream: "tflitetensor_vector_1"
//   input_stream: "tflitetensor_vector_2"
//   output_stream: "concatenated_tflitetensor_vector"
// }
typedef ConcatenateVectorCalculator<TfLiteTensor>
    ConcatenateTfLiteTensorVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateTfLiteTensorVectorCalculator);

typedef ConcatenateVectorCalculator<Tensor> ConcatenateTensorVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateTensorVectorCalculator);

typedef ConcatenateVectorCalculator<::mediapipe::NormalizedLandmark>
    ConcatenateLandmarkVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateLandmarkVectorCalculator);

typedef ConcatenateVectorCalculator<::mediapipe::LandmarkList>
    ConcatenateLandmarkListVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateLandmarkListVectorCalculator);

typedef ConcatenateVectorCalculator<::mediapipe::NormalizedLandmarkList>
    ConcatenateNormalizedLandmarkListVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateNormalizedLandmarkListVectorCalculator);

// For backwards compatibility, keep the version with the typo.
using ConcatenateLandmarListVectorCalculator =
    ConcatenateNormalizedLandmarkListVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateLandmarListVectorCalculator);

typedef ConcatenateVectorCalculator<mediapipe::ClassificationList>
    ConcatenateClassificationListVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateClassificationListVectorCalculator);

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
typedef ConcatenateVectorCalculator<::tflite::gpu::gl::GlBuffer>
    ConcatenateGlBufferVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateGlBufferVectorCalculator);
#endif

typedef ConcatenateVectorCalculator<mediapipe::RenderData>
    ConcatenateRenderDataVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateRenderDataVectorCalculator);

typedef ConcatenateVectorCalculator<mediapipe::Image>
    ConcatenateImageVectorCalculator;
MEDIAPIPE_REGISTER_NODE(ConcatenateImageVectorCalculator);
}  // namespace mediapipe
