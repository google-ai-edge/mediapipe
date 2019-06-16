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

#include "mediapipe/util/tflite/cpu_op_resolver.h"

#include "mediapipe/util/tflite/operations/max_pool_argmax.h"
#include "mediapipe/util/tflite/operations/max_unpooling.h"
#include "mediapipe/util/tflite/operations/transpose_conv_bias.h"
#include "tensorflow/lite/builtin_op_data.h"

namespace mediapipe {

CpuOpResolver::CpuOpResolver() {
  AddCustom("MaxPoolingWithArgmax2D",
            tflite_operations::RegisterMaxPoolingWithArgmax2D());
  AddCustom("MaxUnpooling2D", tflite_operations::RegisterMaxUnpooling2D());
  AddCustom("Convolution2DTransposeBias",
            tflite_operations::RegisterConvolution2DTransposeBias());
}

}  // namespace mediapipe
