// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_PYTHON_PYBIND_MODEL_CKPT_UTIL_H_
#define MEDIAPIPE_PYTHON_PYBIND_MODEL_CKPT_UTIL_H_

#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

void ModelCkptUtilModule(pybind11::module* module);

}  // namespace python
}  // namespace mediapipe

#endif  // MEDIAPIPE_PYTHON_PYBIND_MODEL_CKPT_UTIL_H_
