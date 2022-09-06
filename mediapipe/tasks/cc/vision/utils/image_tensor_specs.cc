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

#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"

#include <stddef.h>

#include <string>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

using ::absl::StatusCode;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::tflite::ColorSpaceType_RGB;
using ::tflite::ContentProperties;
using ::tflite::ContentProperties_ImageProperties;
using ::tflite::EnumNameContentProperties;
using ::tflite::ImageProperties;
using ::tflite::TensorMetadata;
using ::tflite::TensorType;

absl::StatusOr<const ImageProperties*> GetImagePropertiesIfAny(
    const TensorMetadata& tensor_metadata) {
  if (tensor_metadata.content() == nullptr ||
      tensor_metadata.content()->content_properties() == nullptr) {
    return nullptr;
  }

  ContentProperties type = tensor_metadata.content()->content_properties_type();

  if (type != ContentProperties_ImageProperties) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat(
            "Expected ImageProperties for tensor ",
            tensor_metadata.name() ? tensor_metadata.name()->str() : "#0",
            ", got ", EnumNameContentProperties(type), "."),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  return tensor_metadata.content()->content_properties_as_ImageProperties();
}

absl::StatusOr<absl::optional<NormalizationOptions>>
GetNormalizationOptionsIfAny(const TensorMetadata& tensor_metadata) {
  ASSIGN_OR_RETURN(
      const tflite::ProcessUnit* normalization_process_unit,
      ModelMetadataExtractor::FindFirstProcessUnit(
          tensor_metadata, tflite::ProcessUnitOptions_NormalizationOptions));
  if (normalization_process_unit == nullptr) {
    return {absl::nullopt};
  }
  const tflite::NormalizationOptions* tf_normalization_options =
      normalization_process_unit->options_as_NormalizationOptions();
  const auto& mean_values = *tf_normalization_options->mean();
  const auto& std_values = *tf_normalization_options->std();
  if (mean_values.size() != std_values.size()) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("NormalizationOptions: expected mean and std of same "
                     "dimension, got ",
                     mean_values.size(), " and ", std_values.size(), "."),
        MediaPipeTasksStatus::kMetadataInvalidProcessUnitsError);
  }
  absl::optional<NormalizationOptions> normalization_options;
  if (mean_values.size() == 1) {
    normalization_options = NormalizationOptions{
        /* mean_values= */ {mean_values[0], mean_values[0], mean_values[0]},
        /* std_values= */ {std_values[0], std_values[0], std_values[0]},
        /* num_values= */ 1};
  } else if (mean_values.size() == 3) {
    normalization_options = NormalizationOptions{
        /* mean_values= */ {mean_values[0], mean_values[1], mean_values[2]},
        /* std_values= */ {std_values[0], std_values[1], std_values[2]},
        /* num_values= */ 3};
  } else {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("NormalizationOptions: only 1 or 3 mean and std "
                     "values are supported, got ",
                     mean_values.size(), "."),
        MediaPipeTasksStatus::kMetadataInvalidProcessUnitsError);
  }
  return normalization_options;
}

}  // namespace

absl::StatusOr<const TensorMetadata*> GetImageTensorMetadataIfAny(
    const ModelMetadataExtractor& metadata_extractor, int tensor_index) {
  if (metadata_extractor.GetModelMetadata() == nullptr ||
      metadata_extractor.GetModelMetadata()->subgraph_metadata() == nullptr) {
    // Some models have no metadata at all (or very partial), so exit early.
    return nullptr;
  } else if (metadata_extractor.GetInputTensorCount() <= tensor_index) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument, "Tensor index is out of range.",
        MediaPipeTasksStatus::kInvalidNumInputTensorsError);
  }

  const TensorMetadata* metadata =
      metadata_extractor.GetInputTensorMetadata(tensor_index);

  if (metadata == nullptr) {
    // Should never happen.
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "Input TensorMetadata is null.");
  }

  return metadata;
}

absl::StatusOr<ImageTensorSpecs> BuildInputImageTensorSpecs(
    const tflite::Tensor& image_tensor,
    const tflite::TensorMetadata* image_tensor_metadata) {
  const ImageProperties* props = nullptr;
  absl::optional<NormalizationOptions> normalization_options;
  if (image_tensor_metadata != nullptr) {
    ASSIGN_OR_RETURN(props, GetImagePropertiesIfAny(*image_tensor_metadata));
    ASSIGN_OR_RETURN(normalization_options,
                     GetNormalizationOptionsIfAny(*image_tensor_metadata));
  }

  // Input-related specifications.
  if (image_tensor.shape()->size() != 4) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Only 4D tensors in BHWD layout are supported.",
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }
  static constexpr TensorType valid_types[] = {tflite::TensorType_UINT8,
                                               tflite::TensorType_FLOAT32};
  TensorType tensor_type = image_tensor.type();
  if (!absl::c_linear_search(valid_types, tensor_type)) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("Type mismatch for input tensor ",
                     image_tensor.name()->str(),
                     ". Requested one of these types: uint8/float32, got ",
                     tflite::EnumNameTensorType(tensor_type), "."),
        MediaPipeTasksStatus::kInvalidInputTensorTypeError);
  }

  // The expected layout is BHWD, i.e. batch x height x width x color
  // See https://www.tensorflow.org/guide/tensors
  const int* tensor_dims = image_tensor.shape()->data();
  const int batch = tensor_dims[0];
  const int height = tensor_dims[1];
  const int width = tensor_dims[2];
  const int depth = tensor_dims[3];

  if (props != nullptr && props->color_space() != ColorSpaceType_RGB) {
    return CreateStatusWithPayload(StatusCode::kInvalidArgument,
                                   "Only RGB color space is supported for now.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (batch != 1 || depth != 3) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("The input tensor should have dimensions 1 x height x "
                     "width x 3. Got ",
                     batch, " x ", height, " x ", width, " x ", depth, "."),
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }

  size_t byte_depth =
      tensor_type == tflite::TensorType_FLOAT32 ? sizeof(float) : sizeof(uint8);
  int bytes_size = byte_depth * batch * height * width * depth;
  // Sanity checks.
  if (tensor_type == tflite::TensorType_FLOAT32) {
    if (!normalization_options.has_value()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kNotFound,
          "Input tensor has type float32: it requires specifying "
          "NormalizationOptions metadata to preprocess input images.",
          MediaPipeTasksStatus::kMetadataMissingNormalizationOptionsError);
    } else if (bytes_size / sizeof(float) %
                   normalization_options.value().num_values !=
               0) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          "The number of elements in the input tensor must be a multiple of "
          "the number of normalization parameters.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
  }
  if (width <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument, "The input width should be positive.",
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }
  if (height <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument, "The input height should be positive.",
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }

  // Note: in the future, additional checks against `props->default_size()`
  // might be added. Also, verify that NormalizationOptions, if any, do specify
  // a single value when color space is grayscale.
  ImageTensorSpecs result;
  result.image_width = width;
  result.image_height = height;
  result.color_space = ColorSpaceType_RGB;
  result.tensor_type = tensor_type;
  result.normalization_options = normalization_options;

  return result;
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
