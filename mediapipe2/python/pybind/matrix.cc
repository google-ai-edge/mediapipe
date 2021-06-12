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

#include "mediapipe/python/pybind/matrix.h"

#include "mediapipe/framework/formats/matrix.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

void MatrixSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule("matrix", "MediaPipe matrix module.");

  py::class_<mediapipe::Matrix>(m, "Matrix")
      .def(py::init(
          // Pass by reference.
          [](const Eigen::Ref<const Eigen::MatrixXf>& m) { return m; }));
}

}  // namespace python
}  // namespace mediapipe
