#include "mediapipe/calculators/util/landmark_projection_calculator.h"

#include <array>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kImageDimensionsTag[] = "IMAGE_DIMENSIONS";

constexpr float kAbsError = 1e-6;

using ::mediapipe::api3::Graph;
using ::mediapipe::api3::Input;
using ::mediapipe::api3::Output;
using ::mediapipe::api3::Repeated;
using ::mediapipe::api3::Stream;
using ::testing::ElementsAre;

template <typename S>
struct ProjectLandmarksWithRect {
  Repeated<Input<S, mediapipe::NormalizedLandmarkList>> in{"IN"};
  Input<S, NormalizedRect> in_rect{"IN_RECT"};
  Repeated<Output<S, mediapipe::NormalizedLandmarkList>> out{"OUT"};
};

absl::StatusOr<std::vector<mediapipe::NormalizedLandmarkList>>
RunLandmarkProjectionCalculator(std::vector<NormalizedLandmarkList> inputs,
                                NormalizedRect rect) {
  const int num_inputs = inputs.size();
  MP_ASSIGN_OR_RETURN(mediapipe::CalculatorGraphConfig config, [num_inputs]() {
    Graph<ProjectLandmarksWithRect> graph;

    // Graph inputs.
    Stream<NormalizedRect> in_rect_stream = graph.in_rect.Get().SetName("rect");
    std::vector<Stream<NormalizedLandmarkList>> in_streams;
    in_streams.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      in_streams.push_back(graph.in.Add().SetName(absl::StrCat("in", i)));
    }

    // Nodes.
    std::vector<Stream<mediapipe::NormalizedLandmarkList>> output_streams =
        [&]() {
          auto& node = graph.AddNode<api3::LandmarkProjectionNode>();
          node.norm_rect.Set(in_rect_stream);
          std::vector<Stream<mediapipe::NormalizedLandmarkList>> output_streams;
          for (int i = 0; i < in_streams.size(); ++i) {
            node.input_landmarks.Add(in_streams[i]);
            output_streams.push_back(node.output_landmarks.Add());
          }
          return output_streams;
        }();

    // Graph outputs.
    for (int i = 0; i < output_streams.size(); ++i) {
      graph.out.Add(output_streams[i].SetName(absl::StrCat("out", i)));
    }

    return graph.GetConfig();
  }());

  CalculatorGraph calculator_graph;
  RET_CHECK_OK(calculator_graph.Initialize(std::move(config)));

  std::vector<Packet> output_packets;
  output_packets.resize(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    RET_CHECK_OK(calculator_graph.ObserveOutputStream(
        absl::StrCat("out", i), [i, &output_packets](const Packet& p) {
          output_packets[i] = p;
          return absl::OkStatus();
        }));
  }

  RET_CHECK_OK(calculator_graph.StartRun({}));
  RET_CHECK_OK(calculator_graph.AddPacketToInputStream(
      "rect", MakePacket<NormalizedRect>(rect).At(Timestamp(0))));
  for (int i = 0; i < inputs.size(); ++i) {
    RET_CHECK_OK(calculator_graph.AddPacketToInputStream(
        absl::StrCat("in", i),
        MakePacket<NormalizedLandmarkList>(inputs[i]).At(Timestamp(0))));
  }
  RET_CHECK_OK(calculator_graph.CloseAllInputStreams());
  RET_CHECK_OK(calculator_graph.WaitUntilDone());

  std::vector<NormalizedLandmarkList> output_landmarks;
  for (const Packet& p : output_packets) {
    RET_CHECK(!p.IsEmpty());
    output_landmarks.push_back(p.Get<NormalizedLandmarkList>());
  }

  return output_landmarks;
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithDefaultRect) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb");
  mediapipe::NormalizedRect rect =
      ParseTextProtoOrDie<mediapipe::NormalizedRect>(
          R"pb(
            x_center: 0.5,
            y_center: 0.5,
            width: 1.0,
            height: 1.0,
            rotation: 0.0
          )pb");

  MP_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunLandmarkProjectionCalculator({std::move(landmarks)}, std::move(rect)));

  EXPECT_THAT(result,
              ElementsAre(EqualsProto(
                  ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
                    landmark { x: 10, y: 20, z: -0.5 }
                  )pb"))));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingMultipleListsWithDefaultRect) {
  std::vector<mediapipe::NormalizedLandmarkList> landmarks = {
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
        landmark { x: 10, y: 20, z: -0.5 }
        landmark { x: 10, y: 20, z: -0.5 }
      )pb"),
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 20, y: 30, z: 0.5 }
        landmark { x: 20, y: 30, z: 0.5 }
        landmark { x: 20, y: 30, z: 0.5 }
      )pb"),
  };
  mediapipe::NormalizedRect rect =
      ParseTextProtoOrDie<mediapipe::NormalizedRect>(
          R"pb(
            x_center: 0.5,
            y_center: 0.5,
            width: 1.0,
            height: 1.0,
            rotation: 0.0
          )pb");

  MP_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunLandmarkProjectionCalculator(std::move(landmarks), std::move(rect)));

  EXPECT_THAT(
      result,
      ElementsAre(
          EqualsProto(
              ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
                landmark { x: 10, y: 20, z: -0.5 }
                landmark { x: 10, y: 20, z: -0.5 }
                landmark { x: 10, y: 20, z: -0.5 }
              )pb")),
          EqualsProto(
              ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
                landmark { x: 20, y: 30, z: 0.5 }
                landmark { x: 20, y: 30, z: 0.5 }
                landmark { x: 20, y: 30, z: 0.5 }
              )pb"))));
}

mediapipe::NormalizedRect GetCroppedRect() {
  return ParseTextProtoOrDie<mediapipe::NormalizedRect>(
      R"pb(
        x_center: 0.5, y_center: 0.5, width: 0.5, height: 2, rotation: 0.0
      )pb");
}

mediapipe::NormalizedLandmarkList GetCroppedRectTestInput() {
  return ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
    landmark { x: 1.0, y: 1.0, z: -0.5 }
  )pb");
}

mediapipe::NormalizedLandmarkList GetCroppedRectTestExpectedResult() {
  return ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
    landmark { x: 0.75, y: 1.5, z: -0.25 }
  )pb");
}

TEST(LandmarkProjectionCalculatorTest,
     ProjectingWithCroppedRectForSquareImage) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, RunLandmarkProjectionCalculator({GetCroppedRectTestInput()},
                                                   GetCroppedRect()));

  EXPECT_THAT(result,
              ElementsAre(EqualsProto(GetCroppedRectTestExpectedResult())));
}

absl::StatusOr<mediapipe::NormalizedLandmarkList> RunCalculator(
    mediapipe::NormalizedLandmarkList input, mediapipe::NormalizedRect rect,
    std::pair<int, int> image_dimensions) {
  mediapipe::CalculatorRunner runner(
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(R"pb(
        calculator: "LandmarkProjectionCalculator"
        input_stream: "NORM_LANDMARKS:landmarks"
        input_stream: "NORM_RECT:rect"
        input_stream: "IMAGE_DIMENSIONS:image_dimensions"
        output_stream: "NORM_LANDMARKS:projected_landmarks"
      )pb"));
  runner.MutableInputs()
      ->Tag(kNormLandmarksTag)
      .packets.push_back(
          MakePacket<mediapipe::NormalizedLandmarkList>(std::move(input))
              .At(Timestamp(1)));
  runner.MutableInputs()
      ->Tag(kNormRectTag)
      .packets.push_back(MakePacket<mediapipe::NormalizedRect>(std::move(rect))
                             .At(Timestamp(1)));
  runner.MutableInputs()
      ->Tag(kImageDimensionsTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(std::move(image_dimensions))
              .At(Timestamp(1)));

  MP_RETURN_IF_ERROR(runner.Run());
  const auto& output_packets = runner.Outputs().Tag(kNormLandmarksTag).packets;
  RET_CHECK_EQ(output_packets.size(), 1);
  return output_packets[0].Get<mediapipe::NormalizedLandmarkList>();
}

mediapipe::NormalizedRect GetCroppedRectWith90degreeRotation() {
  return ParseTextProtoOrDie<mediapipe::NormalizedRect>(
      R"pb(
        x_center: 0.5,
        y_center: 0.5,
        width: 0.5,
        height: 1,
        rotation: 1.57079632679
      )pb");
}

mediapipe::NormalizedLandmarkList GetCroppedRectTestInputForRotation() {
  return ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
    landmark { x: 0.5, y: 1, z: 0.0 }
  )pb");
}

TEST(LandmarkProjectionCalculatorTest,
     ProjectingWithCroppedRectWithNoRotationForSquareImage) {
  auto status_or_result = RunCalculator(GetCroppedRectTestInput(),
                                        GetCroppedRect(), std::make_pair(1, 1));
  MP_ASSERT_OK(status_or_result);
  auto expected_result = GetCroppedRectTestExpectedResult();
  auto result = status_or_result.value();
  ASSERT_EQ(result.landmark_size(), 1);
  EXPECT_NEAR(result.landmark(0).x(), expected_result.landmark(0).x(),
              kAbsError);
  EXPECT_NEAR(result.landmark(0).y(), expected_result.landmark(0).y(),
              kAbsError);
  EXPECT_NEAR(result.landmark(0).z(), expected_result.landmark(0).z(),
              kAbsError);
}

TEST(LandmarkProjectionCalculatorTest,
     ProjectingWithCroppedRectWithRotationForSquareImage) {
  auto status_or_result =
      RunCalculator(GetCroppedRectTestInputForRotation(),
                    GetCroppedRectWith90degreeRotation(), std::make_pair(1, 1));
  MP_ASSERT_OK(status_or_result);
  auto result = status_or_result.value();
  ASSERT_EQ(result.landmark_size(), 1);
  EXPECT_NEAR(result.landmark(0).x(), 0, kAbsError);
  EXPECT_NEAR(result.landmark(0).y(), 0.5, kAbsError);
  EXPECT_NEAR(result.landmark(0).z(), 0.0, kAbsError);
}

TEST(LandmarkProjectionCalculatorTest,
     ProjectingWithCroppedRectWithRotationForNonSquareImage) {
  auto status_or_result =
      RunCalculator(GetCroppedRectTestInputForRotation(),
                    GetCroppedRectWith90degreeRotation(), std::make_pair(2, 1));
  MP_ASSERT_OK(status_or_result);
  auto result = status_or_result.value();
  ASSERT_EQ(result.landmark_size(), 1);
  EXPECT_NEAR(result.landmark(0).x(), 0.25, kAbsError);
  EXPECT_NEAR(result.landmark(0).y(), 0.5, kAbsError);
  EXPECT_NEAR(result.landmark(0).z(), 0.0, kAbsError);
}

absl::StatusOr<mediapipe::NormalizedLandmarkList> RunCalculator(
    mediapipe::NormalizedLandmarkList input, std::array<float, 16> matrix) {
  mediapipe::CalculatorRunner runner(
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(R"pb(
        calculator: "LandmarkProjectionCalculator"
        input_stream: "NORM_LANDMARKS:landmarks"
        input_stream: "PROJECTION_MATRIX:matrix"
        output_stream: "NORM_LANDMARKS:projected_landmarks"
      )pb"));
  runner.MutableInputs()
      ->Tag(kNormLandmarksTag)
      .packets.push_back(
          MakePacket<mediapipe::NormalizedLandmarkList>(std::move(input))
              .At(Timestamp(1)));
  runner.MutableInputs()
      ->Tag(kProjectionMatrixTag)
      .packets.push_back(MakePacket<std::array<float, 16>>(std::move(matrix))
                             .At(Timestamp(1)));

  MP_RETURN_IF_ERROR(runner.Run());
  const auto& output_packets = runner.Outputs().Tag(kNormLandmarksTag).packets;
  RET_CHECK_EQ(output_packets.size(), 1);
  return output_packets[0].Get<mediapipe::NormalizedLandmarkList>();
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithIdentityMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb");
  // clang-format off
  std::array<float, 16> matrix = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithCroppedRectMatrix) {
  constexpr int kRectWidth = 1280;
  constexpr int kRectHeight = 720;
  auto roi = GetRoi(kRectWidth, kRectHeight, GetCroppedRect());
  std::array<float, 16> matrix;
  GetRotatedSubRectToRectTransformMatrix(roi, kRectWidth, kRectHeight,
                                         /*flip_horizontaly=*/false, &matrix);
  auto status_or_result = RunCalculator(GetCroppedRectTestInput(), matrix);
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(status_or_result.value(),
              EqualsProto(GetCroppedRectTestExpectedResult()));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithScaleMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
        landmark { x: 5, y: 6, z: 7 }
      )pb");
  // clang-format off
  std::array<float, 16> matrix = {
    10.0f, 0.0f,   0.0f, 0.0f,
    0.0f,  100.0f, 0.0f, 0.0f,
    0.0f,  0.0f,   1.0f, 0.0f,
    0.0f,  0.0f,   0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 100, y: 2000, z: -5 }
        landmark { x: 50, y: 600, z: 70 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithTranslateMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb");
  // clang-format off
  std::array<float, 16> matrix = {
    1.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 2.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 11, y: 22, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithRotationMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 4, y: 0, z: -0.5 }
      )pb");
  // clang-format off
  // 90 degrees rotation matrix
  std::array<float, 16> matrix = {
    0.0f, -1.0f, 0.0f, 0.0f,
    1.0f,  0.0f, 0.0f, 0.0f,
    0.0f,  0.0f, 1.0f, 0.0f,
    0.0f,  0.0f, 0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 0, y: 4, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, HasCorrectRegistrationName) {
  EXPECT_EQ(api3::LandmarkProjectionNode::GetRegistrationName(),
            "LandmarkProjectionCalculator");
}

}  // namespace
}  // namespace mediapipe
