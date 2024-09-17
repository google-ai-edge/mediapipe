#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/resources_service.h"

namespace mediapipe {
namespace {

using ::mediapipe::CalculatorGraph;
using ::mediapipe::Timestamp;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;

class TestLabelMapResources : public Resources {
 public:
  explicit TestLabelMapResources(absl::string_view label_map_resource_id,
                                 const std::vector<std::string>& labels)
      : label_map_resource_id_(label_map_resource_id),
        label_map_file_contents_(absl::StrJoin(labels, "\n")) {}

  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const final {
    RET_CHECK_EQ(resource_id, label_map_resource_id_);
    return MakeNoCleanupResource(label_map_file_contents_.data(),
                                 label_map_file_contents_.length());
  }

 private:
  std::string label_map_resource_id_;
  std::string label_map_file_contents_;
};

TEST(DetectionLabelIdToTextCalculator, WorksForLabelMapFile) {
  Graph graph;
  {
    // Graph inputs.
    Stream<std::vector<Detection>> input_detections =
        graph.In(0).Cast<std::vector<Detection>>();
    input_detections.SetName("input_detections");

    // Graph body.
    auto& node = graph.AddNode("DetectionLabelIdToTextCalculator");
    auto& node_opts =
        node.GetOptions<mediapipe::DetectionLabelIdToTextCalculatorOptions>();
    node_opts.set_label_map_path("$LABEL_MAP_PROVIDED_BY_TEST_RESOURCES");
    input_detections.ConnectTo(node.In(0));
    Stream<std::vector<Detection>> detections_with_text_labels =
        node.Out(0).Cast<std::vector<Detection>>();

    // Graph outputs.
    detections_with_text_labels.SetName("detections_with_text_labels");
  }

  // Initialize and run graph.
  CalculatorGraph calculator_graph;
  std::vector<std::string> labels = {"label1", "label2", "label3"};
  std::shared_ptr<Resources> resources =
      std::make_shared<TestLabelMapResources>(
          "$LABEL_MAP_PROVIDED_BY_TEST_RESOURCES", labels);
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(calculator_graph.Initialize(graph.GetConfig()));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(calculator_graph.ObserveOutputStream(
      "detections_with_text_labels", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  // Send input packet.
  std::vector<Detection> input_detections;
  input_detections.resize(labels.size());
  for (int i = 0; i < input_detections.size(); ++i) {
    input_detections[i].add_label_id(i);
    input_detections[i].set_detection_id(i * 1000);
  }
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "input_detections",
      MakePacket<std::vector<Detection>>(input_detections).At(Timestamp(0))));

  // Wait & verify results.
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_EQ(output_packets.size(), 1);
  const Packet& output_packet = output_packets[0];
  ASSERT_FALSE(output_packet.IsEmpty());
  const auto& output_detections = output_packet.Get<std::vector<Detection>>();
  ASSERT_EQ(output_detections.size(), input_detections.size());
  ASSERT_EQ(output_detections.size(), labels.size());
  for (int i = 0; i < output_detections.size(); ++i) {
    EXPECT_EQ(output_detections[i].label_size(), 1);
    EXPECT_EQ(output_detections[i].label(0), labels[i]);
    EXPECT_EQ(output_detections[i].detection_id(),
              input_detections[i].detection_id());
  }

  // Cleanup.
  MP_ASSERT_OK(calculator_graph.CloseAllPacketSources());
  MP_ASSERT_OK(calculator_graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
