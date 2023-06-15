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

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"

namespace mediapipe {
namespace {

constexpr char kMaxPoolingWithArgmax2DOpName[] = "MaxPoolingWithArgmax2D";
constexpr int kMaxPoolingWithArgmax2DOpVersion = 1;

constexpr char kMaxUnpooling2DOpName[] = "MaxUnpooling2D";
constexpr int kMaxUnpooling2DOpVersion = 1;

constexpr char kConvolution2DTransposeBiasOpName[] =
    "Convolution2DTransposeBias";
constexpr int kConvolution2DTransposeBiasOpVersion = 1;

TfLiteRegistration* RegisterMaxPoolingWithArgmax2D() {
  static TfLiteRegistrationExternal* reg_external = []() {
    // Intentionally allocated and never destroyed.
    auto* r = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinCustom, kMaxPoolingWithArgmax2DOpName,
        kMaxPoolingWithArgmax2DOpVersion);
    TfLiteRegistrationExternalSetInit(
        r, [](TfLiteOpaqueContext*, const char*, size_t) -> void* {
          return new TfLitePaddingValues();
        });
    TfLiteRegistrationExternalSetFree(
        r, [](TfLiteOpaqueContext*, void* buffer) -> void {
          delete reinterpret_cast<TfLitePaddingValues*>(buffer);
        });
    TfLiteRegistrationExternalSetPrepare(
        r,
        [](TfLiteOpaqueContext* context,
           TfLiteOpaqueNode* node) -> TfLiteStatus { return kTfLiteOk; });
    TfLiteRegistrationExternalSetInvoke(
        r, [](TfLiteOpaqueContext* context, TfLiteOpaqueNode*) -> TfLiteStatus {
          TfLiteOpaqueContextReportError(
              context, "MaxPoolingWithArgmax2D is only available on the GPU.");
          return kTfLiteError;
        });
    return r;
  }();
  static TfLiteRegistration reg{};
  reg.registration_external = reg_external;
  return &reg;
}

TfLiteRegistration* RegisterMaxUnpooling2D() {
  static TfLiteRegistrationExternal* reg_external =
      // Intentionally allocated and never destroyed.
      TfLiteRegistrationExternalCreate(kTfLiteBuiltinCustom,
                                       kMaxUnpooling2DOpName,
                                       kMaxUnpooling2DOpVersion);
  static TfLiteRegistration reg{};
  reg.registration_external = reg_external;
  return &reg;
}

TfLiteRegistration* RegisterConvolution2DTransposeBias() {
  static TfLiteRegistrationExternal* reg_external =
      // Intentionally allocated and never destroyed.
      TfLiteRegistrationExternalCreate(kTfLiteBuiltinCustom,
                                       kConvolution2DTransposeBiasOpName,
                                       kConvolution2DTransposeBiasOpVersion);
  static TfLiteRegistration reg{};
  reg.registration_external = reg_external;
  return &reg;
}

}  // namespace

OpResolver::OpResolver() {
  AddCustom(kMaxPoolingWithArgmax2DOpName, RegisterMaxPoolingWithArgmax2D(),
            kMaxPoolingWithArgmax2DOpVersion);
  AddCustom(kMaxUnpooling2DOpName, RegisterMaxUnpooling2D(),
            kMaxUnpooling2DOpVersion);
  AddCustom(kConvolution2DTransposeBiasOpName,
            RegisterConvolution2DTransposeBias(),
            kConvolution2DTransposeBiasOpVersion);
}

}  // namespace mediapipe
