// Copyright 2026 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_FROM_IMAGE_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_FROM_IMAGE_CALCULATOR_H_

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#else
#include "mediapipe/framework/api3/any.h"
#endif

namespace mediapipe::api3 {

// A calculator for converting the unified image container into
// legacy MediaPipe datatypes.
//
// Note:
//   Data is automatically transferred to/from the CPU or GPU
//   depending on output type.
//
struct FromImageNode : Node<"FromImageCalculator"> {
  template <typename S>
  struct Contract {
    // Input image.
    Input<S, Image> in_image{"IMAGE"};

    // Output image as ImageFrame.
    Optional<Output<S, ImageFrame>> out_image_cpu{"IMAGE_CPU"};

#if !MEDIAPIPE_DISABLE_GPU
    // Output image as GpuBuffer.
    Optional<Output<S, GpuBuffer>> out_image_gpu{"IMAGE_GPU"};
#else
    Optional<Output<S, Any>> out_image_gpu{"IMAGE_GPU"};
#endif

    // Output indicating whether source image is stored on GPU (or CPU).
    Optional<Output<S, bool>> out_source_on_gpu{"SOURCE_ON_GPU"};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_UTIL_FROM_IMAGE_CALCULATOR_H_
