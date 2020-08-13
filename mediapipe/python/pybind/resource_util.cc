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

#include "mediapipe/python/pybind/resource_util.h"

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"

ABSL_DECLARE_FLAG(std::string, resource_root_dir);

namespace mediapipe {
namespace python {

namespace py = pybind11;

void ResourceUtilSubmodule(pybind11::module* module) {
  py::module m =
      module->def_submodule("resource_util", "MediaPipe resource util module.");

  m.def(
      "set_resource_dir",
      [](const std::string& str) {
        absl::SetFlag(&FLAGS_resource_root_dir, str);
      },
      R"doc(Set resource root directory where can find necessary graph resources such as model files and label maps.

  Args:
    str: A UTF-8 str.

  Examples:
    mp.resource_util.set_resource_dir('/path/to/resource')
)doc");
}

}  // namespace python
}  // namespace mediapipe
