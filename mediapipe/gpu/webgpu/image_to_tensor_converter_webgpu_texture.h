#ifndef MEDIAPIPE_GPU_WEBGPU_IMAGE_TO_TENSOR_CONVERTER_WEBGPU_TEXTURE_H_
#define MEDIAPIPE_GPU_WEBGPU_IMAGE_TO_TENSOR_CONVERTER_WEBGPU_TEXTURE_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {

// Creates image to tensor (represented as WebGPU texture) converter.
// `tensor_type`: float32, uint8 or int8.
// Note: Node::UpdateContract invocation must precede converter creation.
absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateImageToWebGpuTextureTensorConverter(CalculatorContext* cc,
                                          Tensor::ElementType tensor_type);

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_WEBGPU_IMAGE_TO_TENSOR_CONVERTER_WEBGPU_TEXTURE_H_
