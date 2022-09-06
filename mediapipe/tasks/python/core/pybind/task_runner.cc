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

#include "mediapipe/tasks/python/core/pybind/task_runner.h"

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/python/pybind/util.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "pybind11/stl.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace python {

namespace py = pybind11;

namespace {
using ::mediapipe::python::RaisePyErrorIfNotOk;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
using ::mediapipe::tasks::core::TaskRunner;
}  // namespace

// A mutex to guard the python callback function. Only one python callback can
// run at once.
absl::Mutex callback_mutex;

void TaskRunnerSubmodule(py::module* module) {
  pybind11_protobuf::ImportNativeProtoCasters();
  py::module m = module->def_submodule("task_runner",
                                       "MediaPipe Tasks' task runner module.");

  py::class_<TaskRunner> task_runner(m, "TaskRunner",
                                     R"doc(The runner of any MediaPipe Tasks.

TaskRunner is the MediaPipe Tasks core component for running MediaPipe task
graphs. TaskRunner has two processing modes: synchronous mode and asynchronous
mode. In the synchronous mode, clients send input data using the blocking API,
Process(), and wait until the results are returned from the same method. In the
asynchronous mode, clients send input data using the non-blocking method,
Send(), and receive the results in the user-defined packets callback at a later
point in time. As the two processing modes are incompatible, each TaskRunner
instance can operate in only one processing mode, which is defined at
construction time based on whether a packets callback is provided (asynchronous
mode) or not (synchronous mode).)doc");

  task_runner.def_static(
      "create",
      [](CalculatorGraphConfig graph_config,
         std::optional<py::function> packets_callback) {
        PacketsCallback callback = nullptr;
        if (packets_callback.has_value()) {
          callback =
              [packets_callback](absl::StatusOr<PacketMap> output_packets) {
                absl::MutexLock lock(&callback_mutex);
                // Acquires GIL before calling Python callback.
                py::gil_scoped_acquire gil_acquire;
                RaisePyErrorIfNotOk(output_packets.status());
                packets_callback.value()(output_packets.value());
                return absl::OkStatus();
              };
        }
        auto task_runner = TaskRunner::Create(
            std::move(graph_config),
            absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>(),
            std::move(callback));
        RaisePyErrorIfNotOk(task_runner.status());
        return std::move(*task_runner);
      },
      R"doc(Creates a TaskRunner instance from a CalculatorGraphConfig proto and an optional user-defined packets callback.

When a user-defined packets callback is provided, callers must use the
asynchronous method, send(), to provide the input packets. If the packets
callback is absent, clients must use the synchronous method, process(), to
provide the input packets and receive the output packets.

Args:
  graph_config: A MediaPipe task graph config protobuf object.
  packets_callback: A user-defined packets callback function that takes a list
     of output packets as the input argument.

Raises:
  RuntimeError: Any of the following:
    a) The graph config proto is invalid.
    b) The underlying medipaipe graph fails to initilize and start.
)doc",
      py::arg("graph_config"), py::arg("packets_callback") = py::none());

  task_runner.def(
      "process",
      [](TaskRunner* self, const py::dict& input_packets) {
        PacketMap input_packet_map;
        for (const auto& name_to_packet : input_packets) {
          InsertIfNotPresent(&input_packet_map,
                             name_to_packet.first.cast<std::string>(),
                             name_to_packet.second.cast<Packet>());
        }
        py::gil_scoped_release gil_release;
        auto output_packet_map = self->Process(input_packet_map);
        RaisePyErrorIfNotOk(output_packet_map.status(), /**acquire_gil=*/true);
        return std::move(*output_packet_map);
      },
      R"doc(A synchronous method for processing batch data or offline streaming data.

This method is designed for processing either batch data such as unrelated
images and texts or offline streaming data such as the decoded frames from a
video file and an audio file. The call blocks the current thread until a failure
status or a successful result is returned.
If the input packets have no timestamp, an internal timestamp will be assigend
per invocation. Otherwise, when the timestamp is set in the input packets, the
caller must ensure that the input packet timestamps are greater than the
timestamps of the previous invocation. This method is thread-unsafe and it is
the caller's responsibility to synchronize access to this method across multiple
threads and to ensure that the input packet timestamps are in order.

Args:
  input_packets: A dict contains (input stream name, data packet) pairs.

Raises:
  RuntimeError: Any of the following:
    a) TaskRunner is in the asynchronous mode (the packets callback is set).
    b) Any input stream name is not valid.
    c) The underlying medipaipe graph occurs any error during this call.
)doc",
      py::arg("input_packets"));

  task_runner.def(
      "send",
      [](TaskRunner* self, const py::dict& input_packets) {
        PacketMap input_packet_map;
        for (const auto& name_to_packet : input_packets) {
          InsertIfNotPresent(&input_packet_map,
                             name_to_packet.first.cast<std::string>(),
                             name_to_packet.second.cast<Packet>());
        }
        RaisePyErrorIfNotOk(self->Send(input_packet_map));
      },
      R"doc(An asynchronous method for handling live streaming data.

This method that is designed for handling live streaming data such as live
camera and microphone data. A user-defined packets callback function must be
provided in the constructor to receive the output packets. The caller must
ensure that the input packet timestamps are monotonically increasing.
This method is thread-unsafe and it is the caller's responsibility to
synchronize access to this method across multiple threads and to ensure that
the input packet timestamps are in order.

Args:
  input_packets: A dict contains (input stream name, data packet) pairs.

Raises:
  RuntimeError: Any of the following:
    a) TaskRunner is in the synchronous mode (the packets callback is not set).
    b) Any input stream name is not valid.
    c) The packet can't be added into the input stream due to the limited
       queue size or the wrong packet type.
    d) The timestamp of any packet is invalid or is not greater than the
       previously received timestamps.
    e) The underlying medipaipe graph occurs any error during adding input
       packets.
)doc",
      py::arg("input_packets"));

  task_runner.def(
      "close",
      [](TaskRunner* self) {
        py::gil_scoped_release gil_release;
        RaisePyErrorIfNotOk(self->Close(), /**acquire_gil=*/true);
      },
      R"doc(Shuts down the TaskRunner instance.

After the runner is closed, any calls that send input data to the runner are
illegal and will receive errors.

Raises:
  RuntimeError: The underlying medipaipe graph fails to close any input streams
    or calculators.
)doc");

  task_runner.def(
      "restart",
      [](TaskRunner* self) {
        py::gil_scoped_release gil_release;
        RaisePyErrorIfNotOk(self->Restart(), /**acquire_gil=*/true);
      },
      R"doc(Resets and restarts the TaskRunner instance.

This can be useful for resetting a stateful task graph to process new data.

Raises:
  RuntimeError: The underlying medipaipe graph fails to reset and restart.
)doc");
}

}  // namespace python
}  // namespace tasks
}  // namespace mediapipe
