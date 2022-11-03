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

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_IMAGE_PREPROCESSING_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_IMAGE_PREPROCESSING_H_

#include "absl/status/status.h"
#include "mediapipe/tasks/cc/components/image_preprocessing_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"

namespace mediapipe {
namespace tasks {
namespace components {

// Configures an ImagePreprocessing subgraph using the provided model resources
// When use_gpu is true, use GPU as backend to convert image to tensor.
// - Accepts CPU input images and outputs CPU tensors.
//
// Example usage:
//
//   auto& preprocessing =
//       graph.AddNode("mediapipe.tasks.components.ImagePreprocessingSubgraph");
//   core::proto::Acceleration acceleration;
//   acceleration.mutable_xnnpack();
//   bool use_gpu = DetermineImagePreprocessingGpuBackend(acceleration);
//   MP_RETURN_IF_ERROR(ConfigureImagePreprocessing(
//       model_resources,
//       use_gpu,
//       &preprocessing.GetOptions<ImagePreprocessingOptions>()));
//
// The resulting ImagePreprocessing subgraph has the following I/O:
// Inputs:
//   IMAGE - Image
//     The image to preprocess.
//   NORM_RECT - NormalizedRect @Optional
//     Describes region of image to extract.
//     @Optional: rect covering the whole image is used if not specified.
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor populated with the converted and
//     preprocessed image.
//   MATRIX - std::array<float,16> @Optional
//     An std::array<float, 16> representing a 4x4 row-major-order matrix that
//     maps a point on the input image to a point on the output tensor, and
//     can be used to reverse the mapping by inverting the matrix.
//   IMAGE_SIZE - std::pair<int,int> @Optional
//     The size of the original input image as a <width, height> pair.
//   IMAGE - Image @Optional
//     The image that has the pixel data stored on the target storage (CPU vs
//     GPU).
absl::Status ConfigureImagePreprocessing(
    const core::ModelResources& model_resources, bool use_gpu,
    ImagePreprocessingOptions* options);

// Determine if the image preprocessing subgraph should use GPU as the backend
// according to the given acceleration setting.
bool DetermineImagePreprocessingGpuBackend(
    const core::proto::Acceleration& acceleration);

}  // namespace components
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_IMAGE_PREPROCESSING_H_
