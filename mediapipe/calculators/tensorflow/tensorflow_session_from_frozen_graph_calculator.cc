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
//
// Reads serialized GraphDef proto. There are three ways to load a model:
// 1. Specify the path to a graph.pb in the calculator options.
// 2. Specify the path to the graph.pb through the
// input_side_packet:STRING_MODEL_FILE_PATH
// 3. Provide a serialized GraphDef through input_side_packet:STRING_MODEL,
// typically provided by EmbeddingFilePacketFactory.
//
// Produces a SessionBundle that TensorFlowInferenceCalculator can use.

#include <string>

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_frozen_graph_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/deps/monotonic_clock.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/public/session_options.h"

#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

namespace tf = ::tensorflow;

namespace {

constexpr char kSessionTag[] = "SESSION";
constexpr char kStringModelFilePathTag[] = "STRING_MODEL_FILE_PATH";
constexpr char kStringModelTag[] = "STRING_MODEL";

// Updates the graph nodes to use the device as specified by device_id.
void SetPreferredDevice(tf::GraphDef* graph_def, absl::string_view device_id) {
  for (auto& node : *graph_def->mutable_node()) {
    if (node.device().empty()) {
      node.set_device(std::string(device_id));
    }
  }
}
}  // namespace

class TensorFlowSessionFromFrozenGraphCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    const auto& options =
        cc->Options<TensorFlowSessionFromFrozenGraphCalculatorOptions>();
    bool has_exactly_one_model =
        !options.graph_proto_path().empty()
            ? !(cc->InputSidePackets().HasTag(kStringModelTag) |
                cc->InputSidePackets().HasTag(kStringModelFilePathTag))
            : (cc->InputSidePackets().HasTag(kStringModelTag) ^
               cc->InputSidePackets().HasTag(kStringModelFilePathTag));
    RET_CHECK(has_exactly_one_model)
        << "Must have exactly one of graph_proto_path in options or "
           "input_side_packets STRING_MODEL or STRING_MODEL_FILE_PATH";
    if (cc->InputSidePackets().HasTag(kStringModelTag)) {
      cc->InputSidePackets()
          .Tag(kStringModelTag)
          .Set<std::string>(
              // String model from embedded path
          );
    } else if (cc->InputSidePackets().HasTag(kStringModelFilePathTag)) {
      cc->InputSidePackets()
          .Tag(kStringModelFilePathTag)
          .Set<std::string>(
              // Filename of string model.
          );
    }
    cc->OutputSidePackets()
        .Tag(kSessionTag)
        .Set<TensorFlowSession>(
            // A TensorFlow model loaded and ready for use along with
            // a map from tags to tensor names.
        );
    RET_CHECK_GT(options.tag_to_tensor_names().size(), 0);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    auto clock = std::unique_ptr<mediapipe::Clock>(
        mediapipe::MonotonicClock::CreateSynchronizedMonotonicClock());
    const uint64_t start_time = absl::ToUnixMicros(clock->TimeNow());
    const auto& options =
        cc->Options<TensorFlowSessionFromFrozenGraphCalculatorOptions>();
    // Output bundle packet.
    auto session = ::absl::make_unique<TensorFlowSession>();

    tf::SessionOptions session_options;
    session_options.config.CopyFrom(options.config());
    std::vector<mediapipe::ProtoString> initialization_op_names;
    initialization_op_names.reserve(options.initialization_op_names_size());
    for (int i = 0; i < options.initialization_op_names_size(); ++i) {
      initialization_op_names.emplace_back(options.initialization_op_names(i));
    }
    session->session.reset(tf::NewSession(session_options));

    std::string graph_def_serialized;
    if (cc->InputSidePackets().HasTag(kStringModelTag)) {
      graph_def_serialized =
          cc->InputSidePackets().Tag(kStringModelTag).Get<std::string>();
    } else if (cc->InputSidePackets().HasTag(kStringModelFilePathTag)) {
      const std::string& frozen_graph = cc->InputSidePackets()
                                            .Tag(kStringModelFilePathTag)
                                            .Get<std::string>();
      RET_CHECK_OK(
          mediapipe::file::GetContents(frozen_graph, &graph_def_serialized));
    } else {
      RET_CHECK_OK(mediapipe::file::GetContents(options.graph_proto_path(),
                                                &graph_def_serialized));
    }
    tensorflow::GraphDef graph_def;

    RET_CHECK(graph_def.ParseFromString(graph_def_serialized));

    // Update the graph nodes to use the preferred device, if set.
    if (!options.preferred_device_id().empty()) {
      SetPreferredDevice(&graph_def, options.preferred_device_id());
    }

    const tf::Status tf_status = session->session->Create(graph_def);
    RET_CHECK(tf_status.ok()) << "Create failed: " << tf_status.ToString();

    for (const auto& key_value : options.tag_to_tensor_names()) {
      session->tag_to_tensor_map[key_value.first] = key_value.second;
    }
    if (!initialization_op_names.empty()) {
      const tf::Status tf_status =
          session->session->Run({}, {}, initialization_op_names, {});
      // RET_CHECK on the tf::Status object itself in order to print an
      // informative error message.
      RET_CHECK(tf_status.ok()) << "Run failed: " << tf_status.ToString();
    }

    cc->OutputSidePackets().Tag(kSessionTag).Set(Adopt(session.release()));
    const uint64_t end_time = absl::ToUnixMicros(clock->TimeNow());
    ABSL_LOG(INFO) << "Loaded frozen model in: " << end_time - start_time
                   << " microseconds.";
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TensorFlowSessionFromFrozenGraphCalculator);

}  // namespace mediapipe
