/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/audio/utils/audio_tensor_specs.h"

#include <stddef.h>

#include <string>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace {

using ::absl::StatusCode;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::tflite::AudioProperties;
using ::tflite::ContentProperties;
using ::tflite::ContentProperties_AudioProperties;
using ::tflite::EnumNameContentProperties;
using ::tflite::TensorMetadata;
using ::tflite::TensorType;

::absl::StatusOr<const AudioProperties*> GetAudioPropertiesIfAny(
    const TensorMetadata& tensor_metadata) {
  if (tensor_metadata.content() == nullptr ||
      tensor_metadata.content()->content_properties() == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        "Missing audio metadata in the model metadata.",
        MediaPipeTasksStatus::kMetadataNotFoundError);
  }

  ContentProperties type = tensor_metadata.content()->content_properties_type();

  if (type != ContentProperties_AudioProperties) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat(
            "Expected AudioProperties for tensor ",
            tensor_metadata.name() ? tensor_metadata.name()->str() : "#0",
            ", got ", EnumNameContentProperties(type), "."),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  return tensor_metadata.content()->content_properties_as_AudioProperties();
}

}  // namespace

absl::StatusOr<const TensorMetadata*> GetAudioTensorMetadataIfAny(
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

absl::StatusOr<AudioTensorSpecs> BuildInputAudioTensorSpecs(
    const tflite::Tensor& audio_tensor,
    const tflite::TensorMetadata* audio_tensor_metadata) {
  if (audio_tensor_metadata == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        "Missing audio metadata in the model metadata.",
        MediaPipeTasksStatus::kMetadataNotFoundError);
  }

  MP_ASSIGN_OR_RETURN(const AudioProperties* props,
                      GetAudioPropertiesIfAny(*audio_tensor_metadata));
  // Input-related specifications.
  int tensor_shape_size = audio_tensor.shape()->size();
  if (tensor_shape_size > 2) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument, "Only 1D and 2D tensors are supported.",
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }
  static constexpr TensorType valid_types[] = {tflite::TensorType_FLOAT16,
                                               tflite::TensorType_FLOAT32};
  TensorType tensor_type = audio_tensor.type();
  if (!absl::c_linear_search(valid_types, tensor_type)) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("Type mismatch for input tensor ",
                     audio_tensor.name()->str(),
                     ". Requested one of these types: float16/float32, got ",
                     tflite::EnumNameTensorType(tensor_type), "."),
        MediaPipeTasksStatus::kInvalidInputTensorTypeError);
  }

  const int* tensor_dims = audio_tensor.shape()->data();
  int input_buffer_size = 1;
  for (int i = 0; i < tensor_shape_size; i++) {
    if (tensor_dims[i] < 1) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid size: %d for input tensor dimension: %d.",
                          tensor_dims[i], i),
          MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
    }
    input_buffer_size *= tensor_dims[i];
  }

  if (input_buffer_size % props->channels() != 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        absl::StrFormat("Model input tensor size (%d) should be a "
                        "multiplier of the number of channels (%d).",
                        input_buffer_size, props->channels()),
        MediaPipeTasksStatus::kMetadataInconsistencyError);
  }

  AudioTensorSpecs result;
  result.num_channels = props->channels();
  result.num_samples = tensor_dims[tensor_shape_size - 1] / props->channels();
  result.sample_rate = props->sample_rate();
  result.tensor_type = tensor_type;
  result.num_overlapping_samples = 0;

  return result;
}

}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
