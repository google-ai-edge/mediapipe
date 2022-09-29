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

#include "mediapipe/tasks/cc/vision/hand_detector/hand_detector_op_resolver.h"

#include "mediapipe/util/tflite/operations/max_pool_argmax.h"
#include "mediapipe/util/tflite/operations/max_unpooling.h"
#include "mediapipe/util/tflite/operations/transpose_conv_bias.h"

namespace mediapipe {
namespace tasks {
namespace vision {
HandDetectorOpResolver::HandDetectorOpResolver() {
  AddCustom("MaxPoolingWithArgmax2D",
            mediapipe::tflite_operations::RegisterMaxPoolingWithArgmax2D());
  AddCustom("MaxUnpooling2D",
            mediapipe::tflite_operations::RegisterMaxUnpooling2D());
  AddCustom("Convolution2DTransposeBias",
            mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());
}
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
