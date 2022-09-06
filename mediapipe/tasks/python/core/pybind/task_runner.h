// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

#ifndef MEDIAPIPE_TASKS_PYTHON_CORE_PYBIND_TASK_RUNNER_H_
#define MEDIAPIPE_TASKS_PYTHON_CORE_PYBIND_TASK_RUNNER_H_

#include "pybind11/pybind11.h"

namespace mediapipe {
namespace tasks {
namespace python {

void TaskRunnerSubmodule(pybind11::module* module);

}  // namespace python
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_PYTHON_CORE_PYBIND_TASK_RUNNER_H_
