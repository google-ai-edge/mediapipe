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

#include "mediapipe/util/tflite/op_resolver.h"

#include "tensorflow/lite/builtin_op_data.h"

namespace mediapipe {
namespace {

TfLiteRegistration* RegisterMaxPoolingWithArgmax2D() {
  static TfLiteRegistration reg = {
      [](TfLiteContext*, const char*, size_t) -> void* {
        return new TfLitePaddingValues();
      },
      [](TfLiteContext*, void* buffer) -> void {
        delete reinterpret_cast<TfLitePaddingValues*>(buffer);
      },
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return kTfLiteOk;
      },
      [](TfLiteContext* context, TfLiteNode*) -> TfLiteStatus {
        context->ReportError(
            context, "MaxPoolingWithArgmax2D is only available on the GPU.");
        return kTfLiteError;
      },
  };
  return &reg;
}

TfLiteRegistration* RegisterMaxUnpooling2D() {
  static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  return &reg;
}

TfLiteRegistration* RegisterConvolution2DTransposeBias() {
  static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  return &reg;
}

}  // namespace

OpResolver::OpResolver() {
  AddCustom("MaxPoolingWithArgmax2D", RegisterMaxPoolingWithArgmax2D());
  AddCustom("MaxUnpooling2D", RegisterMaxUnpooling2D());
  AddCustom("Convolution2DTransposeBias", RegisterConvolution2DTransposeBias());
}

}  // namespace mediapipe
