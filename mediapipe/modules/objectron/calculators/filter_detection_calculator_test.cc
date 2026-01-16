#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/packet_test_util.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::EqualsProto;

TEST(FilterDetectionCalculatorTest, DetectionFilterTest) {
  auto runner = std::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "FilterDetectionCalculator"
        input_stream: "DETECTION:input"
        output_stream: "DETECTION:output"
        options {
          [mediapipe.FilterDetectionCalculatorOptions.ext]: { min_score: 0.6 }
        }
      )pb"));

  runner->MutableInputs()->Tag("DETECTION").packets = {
      MakePacket<Detection>(ParseTextProtoOrDie<Detection>(R"pb(
        label: "a"
        label: "b"
        label: "c"
        score: 1
        score: 0.8
        score: 0.3
      )pb"))
          .At(Timestamp(20)),
      MakePacket<Detection>(ParseTextProtoOrDie<Detection>(R"pb(
        label: "a"
        label: "b"
        label: "c"
        score: 0.6
        score: 0.4
        score: 0.2
      )pb"))
          .At(Timestamp(40)),
  };

  // Run graph.
  MP_ASSERT_OK(runner->Run());

  // Check output.
  EXPECT_THAT(
      runner->Outputs().Tag("DETECTION").packets,
      ElementsAre(PacketContainsTimestampAndPayload<Detection>(
                      Eq(Timestamp(20)),
                      EqualsProto(R"pb(
                        label: "a" label: "b" score: 1 score: 0.8
                      )pb")),  // Packet 1 at timestamp 20.
                  PacketContainsTimestampAndPayload<Detection>(
                      Eq(Timestamp(40)),
                      EqualsProto(R"pb(
                        label: "a" score: 0.6
                      )pb"))  // Packet 2 at timestamp 40.
                  ));
}

TEST(FilterDetectionCalculatorTest, DetectionFilterLabelMapTest) {
  auto runner = std::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "FilterDetectionCalculator"
        input_stream: "DETECTION:input"
        input_side_packet: "LABEL_MAP:input_map"
        output_stream: "DETECTION:output"
        options {
          [mediapipe.FilterDetectionCalculatorOptions.ext]: { min_score: 0.6 }
        }
      )pb"));

  runner->MutableInputs()->Tag("DETECTION").packets = {
      MakePacket<Detection>(ParseTextProtoOrDie<Detection>(R"pb(
        label: "a"
        label: "b"
        label: "c"
        label: "d"
        score: 1
        score: 0.8
        score: 0.3
        score: 0.9
      )pb"))
          .At(Timestamp(20)),
      MakePacket<Detection>(ParseTextProtoOrDie<Detection>(R"pb(
        label: "a"
        label: "b"
        label: "c"
        label: "e"
        score: 0.6
        score: 0.4
        score: 0.2
        score: 0.7
      )pb"))
          .At(Timestamp(40)),
  };

  auto label_map = std::make_unique<std::map<int, std::string>>();
  (*label_map)[0] = "a";
  (*label_map)[1] = "b";
  (*label_map)[2] = "c";
  runner->MutableSidePackets()->Tag("LABEL_MAP") =
      AdoptAsUniquePtr(label_map.release());

  // Run graph.
  MP_ASSERT_OK(runner->Run());

  // Check output.
  EXPECT_THAT(
      runner->Outputs().Tag("DETECTION").packets,
      ElementsAre(PacketContainsTimestampAndPayload<Detection>(
                      Eq(Timestamp(20)),
                      EqualsProto(R"pb(
                        label: "a" label: "b" score: 1 score: 0.8
                      )pb")),  // Packet 1 at timestamp 20.
                  PacketContainsTimestampAndPayload<Detection>(
                      Eq(Timestamp(40)),
                      EqualsProto(R"pb(
                        label: "a" score: 0.6
                      )pb"))  // Packet 2 at timestamp 40.
                  ));
}

}  // namespace
}  // namespace mediapipe
