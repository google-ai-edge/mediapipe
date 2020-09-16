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

#include "mediapipe/python/pybind/calculator_graph.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/calculator_graph_template.pb.h"
#include "mediapipe/python/pybind/util.h"
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mediapipe {
namespace python {

template <typename T>
T ParseProto(const py::object& proto_object) {
  T proto;
  if (!ParseTextProto<T>(proto_object.str(), &proto)) {
    throw RaisePyError(
        PyExc_RuntimeError,
        absl::StrCat("Failed to parse: ", std::string(proto_object.str()))
            .c_str());
  }
  return proto;
}

namespace py = pybind11;

void CalculatorGraphSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule("calculator_graph",
                                       "MediaPipe calculator graph module.");

  using GraphInputStreamAddMode =
      mediapipe::CalculatorGraph::GraphInputStreamAddMode;

  py::enum_<GraphInputStreamAddMode>(m, "GraphInputStreamAddMode")
      .value("WAIT_TILL_NOT_FULL", GraphInputStreamAddMode::WAIT_TILL_NOT_FULL)
      .value("ADD_IF_NOT_FULL", GraphInputStreamAddMode::ADD_IF_NOT_FULL)
      .export_values();

  // Calculator Graph
  py::class_<CalculatorGraph> calculator_graph(
      m, "CalculatorGraph", R"doc(The primary API for the MediaPipe Framework.

  MediaPipe processing takes place inside a graph, which defines packet flow
  paths between nodes. A graph can have any number of inputs and outputs, and
  data flow can branch and merge. Generally data flows forward, but backward
  loops are possible.)doc");

  // TODO: Support graph initialization with graph templates and
  // subgraph.
  calculator_graph.def(
      py::init([](py::kwargs kwargs) {
        bool init_with_binary_graph = false;
        bool init_with_graph_proto = false;
        bool init_with_validated_graph_config = false;
        CalculatorGraphConfig graph_config_proto;
        for (const auto& kw : kwargs) {
          const std::string& key = kw.first.cast<std::string>();
          if (key == "binary_graph_path") {
            init_with_binary_graph = true;
            std::string file_name(kw.second.cast<py::object>().str());
            graph_config_proto = ReadCalculatorGraphConfigFromFile(file_name);
          } else if (key == "graph_config") {
            init_with_graph_proto = true;
            graph_config_proto =
                ParseProto<CalculatorGraphConfig>(kw.second.cast<py::object>());
          } else if (key == "validated_graph_config") {
            init_with_validated_graph_config = true;
            graph_config_proto =
                py::cast<ValidatedGraphConfig*>(kw.second)->Config();
          } else {
            throw RaisePyError(
                PyExc_RuntimeError,
                absl::StrCat("Unknown kwargs input argument: ", key).c_str());
          }
        }

        if ((init_with_binary_graph ? 1 : 0) + (init_with_graph_proto ? 1 : 0) +
                (init_with_validated_graph_config ? 1 : 0) !=
            1) {
          throw RaisePyError(
              PyExc_ValueError,
              "Please provide \'binary_graph\' to initialize the graph with"
              " binary graph or provide \'graph_config\' to initialize the "
              " with graph config proto or provide \'validated_graph_config\' "
              " to initialize the with ValidatedGraphConfig object.");
        }
        auto calculator_graph = absl::make_unique<CalculatorGraph>();
        RaisePyErrorIfNotOk(calculator_graph->Initialize(graph_config_proto));
        return calculator_graph.release();
      }),
      R"doc(Initialize CalculatorGraph object.

  Args:
    binary_graph_path: The path to a binary mediapipe graph file (.binarypb).
    graph_config: A single CalculatorGraphConfig proto message or its text proto
      format.
    validated_graph_config: A ValidatedGraphConfig object.

  Raises:
    FileNotFoundError: If the binary graph file can't be found.
    ValueError: If the input arguments prvoided are more than needed or the
      graph validation process contains error.
)doc");

  // TODO: Return a Python CalculatorGraphConfig instead.
  calculator_graph.def_property_readonly(
      "text_config",
      [](const CalculatorGraph& self) { return self.Config().DebugString(); });

  calculator_graph.def_property_readonly(
      "binary_config", [](const CalculatorGraph& self) {
        return py::bytes(self.Config().SerializeAsString());
      });

  calculator_graph.def_property_readonly(
      "max_queue_size",
      [](CalculatorGraph* self) { return self->GetMaxInputStreamQueueSize(); });

  calculator_graph.def_property(
      "graph_input_stream_add_mode",
      [](const CalculatorGraph& self) {
        return self.GetGraphInputStreamAddMode();
      },
      [](CalculatorGraph* self, CalculatorGraph::GraphInputStreamAddMode mode) {
        self->SetGraphInputStreamAddMode(mode);
      });

  calculator_graph.def(
      "add_packet_to_input_stream",
      [](CalculatorGraph* self, const std::string& stream, const Packet& packet,
         const Timestamp& timestamp) {
        Timestamp packet_timestamp =
            timestamp == Timestamp::Unset() ? packet.Timestamp() : timestamp;
        if (!packet_timestamp.IsAllowedInStream()) {
          throw RaisePyError(
              PyExc_ValueError,
              absl::StrCat(packet_timestamp.DebugString(),
                           " can't be the timestamp of a Packet in a stream.")
                  .c_str());
        }
        RaisePyErrorIfNotOk(
            self->AddPacketToInputStream(stream, packet.At(packet_timestamp)));
      },
      R"doc(Add a packet to a graph input stream.

  If the graph input stream add mode is ADD_IF_NOT_FULL, the packet will not be
  added if any queue exceeds the max queue size specified by the graph config
  and will raise a Python runtime error. The WAIT_TILL_NOT_FULL mode (default)
  will block until the queues fall below the max queue size before adding the
  packet. If the mode is max queue size is -1, then the packet is added
  regardless of the sizes of the queues in the graph. The input stream must have
  been specified in the configuration as a graph level input stream. On error,
  nothing is added.

  Args:
    stream: The name of the graph input stream.
    packet: The packet to be added into the input stream.
    timestamp: The timestamp of the packet. If set, the original packet
      timestamp will be overwritten.

  Raises:
    RuntimeError: If the stream is not a graph input stream or the packet can't
      be added into the input stream due to the limited queue size or the wrong
      packet type.
    ValueError: If the timestamp of the Packet is invalid to be the timestamp of
      a Packet in a stream.

  Examples:
    graph.add_packet_to_input_stream(
        stream='in',
        packet=packet_creator.create_string('hello world').at(0))

    graph.add_packet_to_input_stream(
        stream='in',
        packet=packet_creator.create_string('hello world'),
        timstamp=1)
)doc",
      py::arg("stream"), py::arg("packet"),
      py::arg("timestamp") = Timestamp::Unset());

  calculator_graph.def(
      "close_input_stream",
      [](CalculatorGraph* self, const std::string& stream) {
        RaisePyErrorIfNotOk(self->CloseInputStream(stream));
      },
      R"doc(Close the named graph input stream.

  Args:
    stream: The name of the stream to be closed.

  Raises:
    RuntimeError: If the stream is not a graph input stream.

)doc");

  calculator_graph.def(
      "close_all_packet_sources",
      [](CalculatorGraph* self) {
        RaisePyErrorIfNotOk(self->CloseAllPacketSources());
      },
      R"doc(Closes all the graph input streams and source calculator nodes.)doc");

  calculator_graph.def(
      "start_run",
      [](CalculatorGraph* self, const pybind11::dict& input_side_packets) {
        std::map<std::string, Packet> input_side_packet_map;
        for (const auto& kv_pair : input_side_packets) {
          InsertIfNotPresent(&input_side_packet_map,
                             kv_pair.first.cast<std::string>(),
                             kv_pair.second.cast<Packet>());
        }
        RaisePyErrorIfNotOk(self->StartRun(input_side_packet_map));
      },

      R"doc(Start a run of the calculator graph.

  A non-blocking call to start a run of the graph and will return when the graph
  is started. If input_side_packets is provided, the method will runs the graph
  after adding the given extra input side packets.

  start_run(), wait_until_done(), has_error(), add_packet_to_input_stream(), and
  close() allow more control over the execution of the graph run.  You can
  insert packets directly into a stream while the graph is running.
  Once start_run() has been called, the graph will continue to run until
  wait_until_done() is called.

  If start_run() returns an error, then the graph is not started and a
  subsequent call to start_run() can be attempted.

  Args:
    input_side_packets: A dict maps from the input side packet names to the
      packets.

  Raises:
    RuntimeError: If the start run occurs any error, e.g. the graph config has
      errors, the calculator can't be found, and the streams are not properly
      connected.

  Examples:
    graph = mp.CalculatorGraph(graph_config=video_process_graph)
    graph.start_run(
        input_side_packets={
            'input_path': packet_creator.create_string('/tmp/input.video'),
            'output_path': packet_creator.create_string('/tmp/output.video')
        })
    graph.close()

    out = []
    graph = mp.CalculatorGraph(graph_config=pass_through_graph)
    graph.observe_output_stream('out',
                                lambda stream_name, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=packet_creator.create_int(0), timestamp=0)
    graph.add_packet_to_input_stream(
        stream='in', packet=packet_creator.create_int(1), timestamp=1)
    graph.close()

)doc",
      py::arg("input_side_packets") = (py::dict){});

  calculator_graph.def(
      "wait_until_done",
      [](CalculatorGraph* self) { RaisePyErrorIfNotOk(self->WaitUntilDone()); },
      R"doc(Wait for the current run to finish.

  A blocking call to wait for the current run to finish (block the current
  thread until all source calculators are stopped, all graph input streams have
  been closed, and no more calculators can be run). This function can be called
  only after start_run(),

  Raises:
    RuntimeError: If the graph occurs any error during the wait call.

  Examples:
    out = []
    graph = mp.CalculatorGraph(graph_config=pass_through_graph)
    graph.observe_output_stream('out', lambda stream_name, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=packet_creator.create_int(0), timestamp=0)
    graph.close_all_packet_sources()
    graph.wait_until_done()

)doc");

  calculator_graph.def(
      "wait_until_idle",
      [](CalculatorGraph* self) { RaisePyErrorIfNotOk(self->WaitUntilIdle()); },
      R"doc(Wait until the running graph is in the idle mode.

  Wait until the running graph is in the idle mode, which is when nothing can
  be scheduled and nothing is running in the worker threads. This function can
  be called only after start_run().

  NOTE: The graph must not have any source nodes because source nodes prevent
  the running graph from becoming idle until the source nodes are done.

  Raises:
    RuntimeError: If the graph occurs any error during the wait call.

  Examples:
    out = []
    graph = mp.CalculatorGraph(graph_config=pass_through_graph)
    graph.observe_output_stream('out',
                                lambda stream_name, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=packet_creator.create_int(0), timestamp=0)
    graph.wait_until_idle()

)doc");

  calculator_graph.def(
      "wait_for_observed_output",
      [](CalculatorGraph* self) {
        RaisePyErrorIfNotOk(self->WaitForObservedOutput());
      },
      R"doc(Wait until a packet is emitted on one of the observed output streams.

  Returns immediately if a packet has already been emitted since the last
  call to this function.

  Raises:
    RuntimeError:
      If the graph occurs any error or the graph is terminated while waiting.

  Examples:
    out = []
    graph = mp.CalculatorGraph(graph_config=pass_through_graph)
    graph.observe_output_stream('out',
                                lambda stream_name, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=packet_creator.create_int(0), timestamp=0)
    graph.wait_for_observed_output()
    value = packet_getter.get_int(out[0])
    graph.add_packet_to_input_stream(
        stream='in', packet=packet_creator.create_int(1), timestamp=1)
    graph.wait_for_observed_output()
    value = packet_getter.get_int(out[1])

)doc");

  calculator_graph.def(
      "has_error", [](const CalculatorGraph& self) { return self.HasError(); },
      R"doc(Quick non-locking means of checking if the graph has encountered an error)doc");

  calculator_graph.def(
      "get_combined_error_message",
      [](CalculatorGraph* self) {
        ::mediapipe::Status error_status;
        if (self->GetCombinedErrors(&error_status) && !error_status.ok()) {
          return error_status.ToString();
        }
        return std::string();
      },
      R"doc(Combines error messages as a single std::string.

  Examples:
    if graph.has_error():
      print(graph.get_combined_error_message())

)doc");

  // TODO: Support passing a single-argument lambda for convenience.
  calculator_graph.def(
      "observe_output_stream",
      [](CalculatorGraph* self, const std::string& stream_name,
         pybind11::function callback_fn) {
        RaisePyErrorIfNotOk(self->ObserveOutputStream(
            stream_name, [callback_fn, stream_name](const Packet& packet) {
              callback_fn(stream_name, packet);
              return mediapipe::OkStatus();
            }));
      },
      R"doc(Observe the named output stream.

  callback_fn will be invoked on every packet emitted by the output stream.
  This method can only be called before start_run().

  Args:
    stream_name: The name of the output stream.
    callback_fn: The callback function to invoke on every packet emitted by the
      output stream.

  Raises:
    RuntimeError: If the calculator graph isn't initialized or the stream
      doesn't exist.

  Examples:
    out = []
    graph = mp.CalculatorGraph(graph_config=graph_config)
    graph.observe_output_stream('out',
                                lambda stream_name, packet: out.append(packet))

)doc");

  calculator_graph.def(
      "close",
      [](CalculatorGraph* self) {
        RaisePyErrorIfNotOk(self->CloseAllPacketSources());
        RaisePyErrorIfNotOk(self->WaitUntilDone());
      },
      R"doc(Close all the input sources and shutdown the graph.)doc");

  calculator_graph.def(
      "get_output_side_packet",
      [](CalculatorGraph* self, const std::string& packet_name) {
        auto status_or_packet = self->GetOutputSidePacket(packet_name);
        RaisePyErrorIfNotOk(status_or_packet.status());
        return status_or_packet.ValueOrDie();
      },
      R"doc(Get output side packet by name after the graph is done.

  Args:
    stream: The name of the outnput stream.

  Raises:
    RuntimeError: If the graph is still running or the output side packet is not
      found or empty.

  Examples:
    graph = mp.CalculatorGraph(graph_config=graph_config)
    graph.start_run()
    graph.close()
    output_side_packet = graph.get_output_side_packet('packet_name')

)doc",
      py::return_value_policy::move);
}

}  // namespace python
}  // namespace mediapipe
