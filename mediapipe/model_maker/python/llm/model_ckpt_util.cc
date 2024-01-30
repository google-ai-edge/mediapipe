// Copyright 2024 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_TASKS_TOOLS_MODEL_CKPT_UTIL_H_
#define MEDIAPIPE_TASKS_TOOLS_MODEL_CKPT_UTIL_H_

#include "odml/infra/genai/inference/utils/xnn_utils/model_ckpt_util.h"

#include "odml/infra/genai/inference/ml_drift/llm/tensor_loaders/model_ckpt_util.h"
#include "pybind11/pybind11.h"
#include "third_party/pybind11_abseil/status_casters.h"

PYBIND11_MODULE(model_ckpt_util, m) {
  pybind11::google::ImportStatusModule();
  m.doc() = "Pybind model checkpoint utility functions.";

  m.def("GenerateXnnpackTfLite", &odml::infra::xnn_utils::GenerateTfLite,
        "Generates the TfLite flatbuffer file from the serialized weight files "
        "for XNNPACK.");
  m.def("GenerateMlDriftTfLite", &odml::infra::gpu::GenerateTfLite,
        "Generates the TfLite flatbuffer file from the serialized weight files "
        "for ML Drift.");
}

#endif  // MEDIAPIPE_TASKS_TOOLS_MODEL_CKPT_UTIL_H_
