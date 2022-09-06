// Copyright 2020-2021 The MediaPipe Authors.
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

#include "mediapipe/python/pybind/calculator_graph.h"
#include "mediapipe/python/pybind/image.h"
#include "mediapipe/python/pybind/image_frame.h"
#include "mediapipe/python/pybind/matrix.h"
#include "mediapipe/python/pybind/packet.h"
#include "mediapipe/python/pybind/packet_creator.h"
#include "mediapipe/python/pybind/packet_getter.h"
#include "mediapipe/python/pybind/resource_util.h"
#include "mediapipe/python/pybind/timestamp.h"
#include "mediapipe/python/pybind/validated_graph_config.h"
#include "mediapipe/tasks/python/core/pybind/task_runner.h"

namespace mediapipe {
namespace python {

PYBIND11_MODULE(_framework_bindings, m) {
  ResourceUtilSubmodule(&m);
  ImageSubmodule(&m);
  ImageFrameSubmodule(&m);
  MatrixSubmodule(&m);
  TimestampSubmodule(&m);
  PacketSubmodule(&m);
  PacketCreatorSubmodule(&m);
  PacketGetterSubmodule(&m);
  CalculatorGraphSubmodule(&m);
  ValidatedGraphConfigSubmodule(&m);
  // As all MediaPipe calculators and Python bindings need to go into a single
  // .so file, having MediaPipe Tasks' task runner module in _framework_bindings
  // as well.
  tasks::python::TaskRunnerSubmodule(&m);
}

}  // namespace python
}  // namespace mediapipe
