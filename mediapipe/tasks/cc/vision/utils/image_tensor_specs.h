/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_IMAGE_TENSOR_SPECS_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_IMAGE_TENSOR_SPECS_H_

#include <array>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {

// Parameters used for input image normalization when input tensor has
// kTfLiteFloat32 type.
//
// Exactly 1 or 3 values are expected for `mean_values` and `std_values`. In
// case 1 value only is specified, it is used for all channels. E.g. for a RGB
// image, the normalization is done as follow:
//
//   (R - mean_values[0]) / std_values[0]
//   (G - mean_values[1]) / std_values[1]
//   (B - mean_values[2]) / std_values[2]
//
// `num_values` keeps track of how many values have been provided, which should
// be 1 or 3 (see above). In particular, single-channel grayscale images expect
// only 1 value.
struct NormalizationOptions {
  std::array<float, 3> mean_values;
  std::array<float, 3> std_values;
  int num_values;
};

// Parameters related to the expected tensor specifications when the tensor
// represents an image.
//
// E.g. Before running inference with the TF Lite interpreter, the caller must
// use these values and perform image preprocessing and/or normalization so as
// to fill the actual input tensor appropriately.
struct ImageTensorSpecs {
  // Expected image dimensions, e.g. image_width=224, image_height=224.
  int image_width;
  int image_height;
  // Expected color space, e.g. color_space=RGB.
  tflite::ColorSpaceType color_space;
  // Expected input tensor type, e.g. if tensor_type=TensorType_FLOAT32 the
  // caller should usually perform some normalization to convert the uint8
  // pixels into floats (see NormalizationOptions in TF Lite Metadata for more
  // details).
  tflite::TensorType tensor_type;
  // Optional normalization parameters read from TF Lite Metadata. Those are
  // mandatory when tensor_type=TensorType_FLOAT32 in order to convert the input
  // image data into the expected range of floating point values, an error is
  // returned otherwise (see sanity checks below). They should be ignored for
  // other tensor input types, e.g. kTfLiteUInt8.
  absl::optional<NormalizationOptions> normalization_options;
};

// Gets the image tensor metadata from the metadata extractor by tensor index.
absl::StatusOr<const tflite::TensorMetadata*> GetImageTensorMetadataIfAny(
    const metadata::ModelMetadataExtractor& metadata_extractor,
    int tensor_index);

// Performs sanity checks on the expected input tensor including consistency
// checks against model metadata, if any. For now, a single RGB input with BHWD
// layout, where B = 1 and D = 3, is expected. Returns the corresponding input
// specifications if they pass, or an error otherwise (too many input tensors,
// etc).
// Note: both model and metadata extractor *must* be successfully
// initialized before calling this function by means of (respectively):
// - `tflite::GetModel`,
// - `mediapipe::metadata::ModelMetadataExtractor::CreateFromModelBuffer`.
absl::StatusOr<ImageTensorSpecs> BuildInputImageTensorSpecs(
    const tflite::Tensor& image_tensor,
    const tflite::TensorMetadata* image_tensor_metadata);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_IMAGE_TENSOR_SPECS_H_
