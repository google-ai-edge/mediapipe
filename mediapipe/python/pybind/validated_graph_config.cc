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

#include "mediapipe/python/pybind/validated_graph_config.h"

#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/validated_graph_config.h"
#include "mediapipe/python/pybind/util.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

void ValidatedGraphConfigSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule(
      "validated_graph_config", "MediaPipe validated graph config module.");

  // Validated Graph Config
  py::class_<ValidatedGraphConfig> validated_graph_config(
      m, "ValidatedGraphConfig",
      R"doc(A class to validate and canonicalize a CalculatorGraphConfig.)doc");

  validated_graph_config.def(py::init())
      .def(
          "initialize",
          [](ValidatedGraphConfig* self, py::kwargs kwargs) {
            bool init_with_binary_graph = false;
            bool init_with_graph_proto = false;
            CalculatorGraphConfig graph_config_proto;
            for (const auto& kw : kwargs) {
              const std::string& key = kw.first.cast<std::string>();
              if (key == "binary_graph_path") {
                init_with_binary_graph = true;
                std::string file_name(kw.second.cast<py::object>().str());
                graph_config_proto =
                    ReadCalculatorGraphConfigFromFile(file_name);
              } else if (key == "graph_config") {
                init_with_graph_proto = true;
                if (!ParseTextProto<CalculatorGraphConfig>(
                        kw.second.cast<py::object>().str(),
                        &graph_config_proto)) {
                  throw RaisePyError(
                      PyExc_RuntimeError,
                      absl::StrCat(
                          "Failed to parse: ",
                          std::string(kw.second.cast<py::object>().str()))
                          .c_str());
                }
              } else {
                throw RaisePyError(
                    PyExc_RuntimeError,
                    absl::StrCat("Unknown kwargs input argument: ", key)
                        .c_str());
              }
            }
            if (!(init_with_binary_graph ^ init_with_graph_proto)) {
              throw RaisePyError(
                  PyExc_ValueError,
                  "Please either provide \'binary_graph_path\' to initialize "
                  "a ValidatedGraphConfig object with a binary graph file or "
                  "\'graph_config\' to initialize a ValidatedGraphConfig "
                  "object with a graph config proto.");
            }
            RaisePyErrorIfNotOk(self->Initialize(graph_config_proto));
          },
          R"doc(Initialize ValidatedGraphConfig with a CalculatorGraphConfig.

  Args:
    binary_graph_path: The path to a binary mediapipe graph file (.binarypb).
    graph_config: A single CalculatorGraphConfig proto message or its text proto
      format.

  Raises:
    FileNotFoundError: If the binary graph file can't be found.
    ValueError: If the input arguments prvoided are more than needed or the
      graph validation process contains error.

  Examples:
    validated_graph_config = mp.ValidatedGraphConfig()
    validated_graph_config.initialize(graph_config=text_config)

)doc");

  validated_graph_config.def(
      "registered_stream_type_name",
      [](ValidatedGraphConfig& self, const std::string& stream_name) {
        auto status_or_type_name = self.RegisteredStreamTypeName(stream_name);
        RaisePyErrorIfNotOk(status_or_type_name.status());
        return status_or_type_name.value();
      },
      R"doc(Return the registered type name of the specified stream if it can be determined.

  Args:
    stream_name: The input/output stream name.

  Returns:
    The registered packet type name of the input/output stream.

  Raises:
    ValueError: If the input/output stream cannot be found.

  Examples:
    validated_graph_config.registered_stream_type_name('stream_name')

)doc");

  validated_graph_config.def(
      "registered_side_packet_type_name",
      [](ValidatedGraphConfig& self, const std::string& side_packet_name) {
        auto status_or_type_name =
            self.RegisteredSidePacketTypeName(side_packet_name);
        RaisePyErrorIfNotOk(status_or_type_name.status());
        return status_or_type_name.value();
      },
      R"doc(Return the registered type name of the specified side packet if it can be determined.

  Args:
    side_packet_name: The input/output side packet name.

  Returns:
    The registered packet type name of the input/output side packet.

  Raises:
    ValueError: If the input/output side packet cannot be found.

  Examples:
    validated_graph_config.registered_side_packet_type_name('side_packet')

)doc");

  // TODO: Return a Python CalculatorGraphConfig instead.
  validated_graph_config.def_property_readonly(
      "text_config", [](const ValidatedGraphConfig& self) {
        return self.Config().DebugString();
      });

  validated_graph_config.def_property_readonly(
      "binary_config", [](const ValidatedGraphConfig& self) {
        return py::bytes(self.Config().SerializeAsString());
      });

  validated_graph_config.def(
      "initialized",
      [](const ValidatedGraphConfig& self) { return self.Initialized(); },
      R"doc(Indicate if ValidatedGraphConfig is initialized.)doc");
}

}  // namespace python
}  // namespace mediapipe
