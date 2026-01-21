// Copyright 2025 The MediaPipe Authors.
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
#ifndef MEDIAPIPE_GPU_IMAGE_FRAME_TO_GPU_BUFFER_CALCULATOR_H_
#define MEDIAPIPE_GPU_IMAGE_FRAME_TO_GPU_BUFFER_CALCULATOR_H_

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif

namespace mediapipe::api3 {

// Convert ImageFrame to GpuBuffer.
//
// NOTE: all ImageFrameToGpuBufferCalculators use a common dedicated shared GL
// context thread by default, which is different from the main GL context thread
// used by the graph. (If MediaPipe uses multithreading and multiple OpenGL
// contexts.)
struct ImageFrameToGpuBufferNode : Node<"ImageFrameToGpuBufferCalculator"> {
  template <typename S>
  struct Contract {
    Input<S, ImageFrame> image_frame{""};
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    Optional<SideInput<S, GpuSharedData*>> gpu_shared{"GPU_SHARED"};
#endif
    Output<S, GpuBuffer> gpu_buffer{""};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_GPU_IMAGE_FRAME_TO_GPU_BUFFER_CALCULATOR_H_
