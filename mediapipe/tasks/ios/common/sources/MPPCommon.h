// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum MPPTasksErrorCode
 * This enum specifies  error codes for Mediapipe Task Library.
 * It maintains a 1:1 mapping to MediaPipeTasksStatus of the C ++libray.
 */
typedef NS_ENUM(NSUInteger, MPPTasksErrorCode) {

  // Generic error codes.

  // Unspecified error.
  MPPTasksErrorCodeError = 1,
  // Invalid argument specified.
  MPPTasksErrorCodeInvalidArgumentError = 2,
  // Invalid FlatBuffer file or buffer specified.
  MPPTasksErrorCodeInvalidFlatBufferError = 3,
  // Model contains a builtin op that isn't supported by the OpResolver or
  // delegates.
  MPPTasksErrorCodeUnsupportedBuiltinOp = 4,
  // Model contains a custom op that isn't supported by the OpResolver or
  // delegates.
  MPPTasksErrorCodeUnsupportedCustomOp = 5,

  // File I/O error codes.

  // No such file.
  MPPTasksErrorCodeFileNotFoundError = 100,
  // Permission issue.
  MPPTasksErrorCodeFilePermissionDeniedError,
  // I/O error when reading file.
  MPPTasksErrorCodeFileReadError,
  // I/O error when mmap-ing file.
  MPPTasksErrorCodeFileMmapError,
  // ZIP I/O error when unpacking the zip file.
  MPPTasksErrorCodeFileZipError,

  // TensorFlow Lite metadata error codes.

  // Unexpected schema version (aka file_identifier) in the Metadata FlatBuffer.
  MPPTasksErrorCodeMetadataInvalidSchemaVersionError = 200,
  // No such associated file within metadata, or file has not been packed.
  MPPTasksErrorCodeMetadataAssociatedFileNotFoundError,
  // ZIP I/O error when unpacking an associated file.
  MPPTasksErrorCodeMetadataAssociatedFileZipError,
  // Inconsistency error between the metadata and actual TF Lite model.
  // E.g.: number of labels and output tensor values differ.
  MPPTasksErrorCodeMetadataInconsistencyError,
  // Invalid process units specified.
  // E.g.: multiple ProcessUnits with the same type for a given tensor.
  MPPTasksErrorCodeMetadataInvalidProcessUnitsError,
  // Inconsistency error with the number of labels.
  // E.g.: label files for different locales have a different number of labels.
  MPPTasksErrorCodeMetadataNumLabelsMismatchError,
  // Score calibration parameters parsing error.
  // E.g.: too many parameters provided in the corresponding associated file.
  MPPTasksErrorCodeMetadataMalformedScoreCalibrationError,
  // Unexpected number of subgraphs for the current task.
  // E.g.: image classification expects a single subgraph.
  MPPTasksErrorCodeMetadataInvalidNumSubgraphsError,
  // A given tensor requires NormalizationOptions but none were found.
  // E.g.: float input tensor requires normalization to preprocess input images.
  MPPTasksErrorCodeMetadataMissingNormalizationOptionsError,
  // Invalid ContentProperties specified.
  // E.g. expected ImageProperties, got BoundingBoxProperties.
  MPPTasksErrorCodeMetadataInvalidContentPropertiesError,
  // Metadata is mandatory but was not found.
  // E.g. current task requires TFLite Model Metadata but none was found.
  MPPTasksErrorCodeMetadataNotFoundError,
  // Associated TENSOR_AXIS_LABELS or TENSOR_VALUE_LABELS file is mandatory but
  // none was found or it was empty.
  // E.g. current task requires labels but none were found.
  MPPTasksErrorCodeMetadataMissingLabelsError,
  // The ProcessingUnit for tokenizer is not correctly configured.
  // E.g BertTokenizer doesn't have a valid vocab file associated.
  MPPTasksErrorCodeMetadataInvalidTokenizerError,

  // Input tensor(s) error codes.

  // Unexpected number of input tensors for the current task.
  // E.g. current task expects a single input tensor.
  MPPTasksErrorCodeInvalidNumInputTensorsError = 300,
  // Unexpected input tensor dimensions for the current task.
  // E.g.: only 4D input tensors supported.
  MPPTasksErrorCodeInvalidInputTensorDimensionsError,
  // Unexpected input tensor type for the current task.
  // E.g.: current task expects a uint8 pixel image as input.
  MPPTasksErrorCodeInvalidInputTensorTypeError,
  // Unexpected input tensor bytes size.
  // E.g.: size in bytes does not correspond to the expected number of pixels.
  MPPTasksErrorCodeInvalidInputTensorSizeError,
  // No correct input tensor found for the model.
  // E.g.: input tensor name is not part of the text model's input tensors.
  MPPTasksErrorCodeInputTensorNotFoundError,

  // Output tensor(s) error codes.

  // Unexpected output tensor dimensions for the current task.
  // E.g.: only a batch size of 1 is supported.
  MPPTasksErrorCodeInvalidOutputTensorDimensionsError = 400,
  // Unexpected input tensor type for the current task.
  // E.g.: multi-head model with different output tensor types.
  MPPTasksErrorCodeInvalidOutputTensorTypeError,
  // No correct output tensor found for the model.
  // E.g.: output tensor name is not part of the text model's output tensors.
  MPPTasksErrorCodeOutputTensorNotFoundError,
  // Unexpected number of output tensors for the current task.
  // E.g.: current task expects a single output tensor.
  MPPTasksErrorCodeInvalidNumOutputTensorsError,

  // Image processing error codes.

  // Unspecified image processing failures.
  MPPTasksErrorCodeImageProcessingError = 500,
  // Unexpected input or output buffer metadata.
  // E.g.: rotate RGBA buffer to Grayscale buffer by 90 degrees.
  MPPTasksErrorCodeImageProcessingInvalidArgumentError,
  // Image processing operation failures.
  // E.g. libyuv rotation failed for an unknown reason.
  MPPTasksErrorCodeImageProcessingBackendError,

  // Task runner error codes.
  MPPTasksErrorCodeRunnerError = 600,
  // Task runner is not initialized.
  MPPTasksErrorCodeRunnerInitializationError,
  // Task runner is not started successfully.
  MPPTasksErrorCodeRunnerFailsToStartError,
  // Task runner is not started.
  MPPTasksErrorCodeRunnerNotStartedError,
  // Task runner API is called in the wrong processing mode.
  MPPTasksErrorCodeRunnerApiCalledInWrongModeError,
  // Task runner receives/produces invalid MediaPipe packet timestamp.
  MPPTasksErrorCodeRunnerInvalidTimestampError,
  // Task runner receives unexpected MediaPipe graph input packet.
  // E.g. The packet type doesn't match the graph input stream's data type.
  MPPTasksErrorCodeRunnerUnexpectedInputError,
  // Task runner produces unexpected MediaPipe graph output packet.
  // E.g. The number of output packets is not equal to the number of graph
  // output streams.
  MPPTasksErrorCodeRunnerUnexpectedOutputError,
  // Task runner is not closed successfully.
  MPPTasksErrorCodeRunnerFailsToCloseError,
  // Task runner's model resources cache service is unavailable or the
  // targeting model resources bundle is not found.
  MPPTasksErrorCodeRunnerModelResourcesCacheServiceError,

  // Task graph error codes.
  MPPTasksErrorCodeGraphError = 700,
  // Task graph is not implemented.
  MPPTasksErrorCodeTaskGraphNotImplementedError,
  // Task graph config is invalid.
  MPPTasksErrorCodeInvalidTaskGraphConfigError,

  // The first error code in MPPTasksErrorCode (for internal use only).
  MPPTasksErrorCodeFirst = MPPTasksErrorCodeError,

  // The last error code in MPPTasksErrorCode (for internal use only).
  MPPTasksErrorCodeLast = MPPTasksErrorCodeInvalidTaskGraphConfigError,

} NS_SWIFT_NAME(TasksErrorCode);

NS_ASSUME_NONNULL_END
