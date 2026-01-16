#include "mediapipe/calculators/util/landmark_projection_calculator.h"

#include <array>
#include <tuple>
#include <utility>

#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

constexpr float kAbsError = 1e-6;

using ::testing::ElementsAre;

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
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<NormalizedRect> rect)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.norm_rect.Set(rect);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> output,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(
                     ParseTextProtoOrDie<NormalizedLandmarkList>(R"pb(
                       landmark { x: 10, y: 20, z: -0.5 }
                     )pb")),
                 api3::MakePacket<NormalizedRect>(
                     ParseTextProtoOrDie<NormalizedRect>(R"pb(
                       x_center: 0.5,
                       y_center: 0.5,
                       width: 1.0,
                       height: 1.0,
                       rotation: 0.0
                     )pb"))));
  ASSERT_TRUE(output);
  const NormalizedLandmarkList& result = output.GetOrDie();

  EXPECT_THAT(
      result,
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingMultipleListsWithDefaultRect) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph,
                     Stream<NormalizedLandmarkList> landmarks_a,
                     Stream<NormalizedLandmarkList> landmarks_b,
                     Stream<NormalizedRect> rect)
                      -> std::tuple<Stream<NormalizedLandmarkList>,
                                    Stream<NormalizedLandmarkList>> {
        auto& node = graph.AddNode<LandmarkProjectionNode>();
        node.norm_rect.Set(rect);
        node.input_landmarks.Add(landmarks_a);
        node.input_landmarks.Add(landmarks_b);
        return {node.output_landmarks.Add(), node.output_landmarks.Add()};
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      (auto [output_a, output_b]),
      runner.Run(
          api3::MakePacket<NormalizedLandmarkList>(
              ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
                landmark { x: 10, y: 20, z: -0.5 }
                landmark { x: 10, y: 20, z: -0.5 }
                landmark { x: 10, y: 20, z: -0.5 }
              )pb")),
          api3::MakePacket<NormalizedLandmarkList>(
              ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
                landmark { x: 20, y: 30, z: 0.5 }
                landmark { x: 20, y: 30, z: 0.5 }
                landmark { x: 20, y: 30, z: 0.5 }
              )pb")),
          api3::MakePacket<NormalizedRect>(
              ParseTextProtoOrDie<mediapipe::NormalizedRect>(
                  R"pb(
                    x_center: 0.5,
                    y_center: 0.5,
                    width: 1.0,
                    height: 1.0,
                    rotation: 0.0
                  )pb"))));
  ASSERT_TRUE(output_a);
  ASSERT_TRUE(output_b);
  auto result = {output_a.GetOrDie(), output_b.GetOrDie()};

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
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<NormalizedRect> rect)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.norm_rect.Set(rect);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> output,
      runner.Run(
          api3::MakePacket<NormalizedLandmarkList>(GetCroppedRectTestInput()),
          api3::MakePacket<NormalizedRect>(GetCroppedRect())));
  ASSERT_TRUE(output);
  EXPECT_THAT(output.GetOrDie(),
              EqualsProto(GetCroppedRectTestExpectedResult()));
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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<NormalizedRect> norm_rect,
                                  Stream<std::pair<int, int>> dimensions)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.norm_rect.Set(norm_rect);
                     node.image_dimensions.Set(dimensions);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> output,
      runner.Run(
          api3::MakePacket<NormalizedLandmarkList>(GetCroppedRectTestInput()),
          api3::MakePacket<NormalizedRect>(GetCroppedRect()),
          api3::MakePacket<std::pair<int, int>>(1, 1)));
  ASSERT_TRUE(output);
  const auto& result = output.GetOrDie();
  const auto& expected_result = GetCroppedRectTestExpectedResult();

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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<NormalizedRect> norm_rect,
                                  Stream<std::pair<int, int>> dimensions)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.norm_rect.Set(norm_rect);
                     node.image_dimensions.Set(dimensions);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> output,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(
                     GetCroppedRectTestInputForRotation()),
                 api3::MakePacket<NormalizedRect>(
                     GetCroppedRectWith90degreeRotation()),
                 api3::MakePacket<std::pair<int, int>>(1, 1)));
  ASSERT_TRUE(output);
  const auto& result = output.GetOrDie();
  ASSERT_EQ(result.landmark_size(), 1);
  EXPECT_NEAR(result.landmark(0).x(), 0, kAbsError);
  EXPECT_NEAR(result.landmark(0).y(), 0.5, kAbsError);
  EXPECT_NEAR(result.landmark(0).z(), 0.0, kAbsError);
}

TEST(LandmarkProjectionCalculatorTest,
     ProjectingWithCroppedRectWithRotationForNonSquareImage) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<NormalizedRect> norm_rect,
                                  Stream<std::pair<int, int>> dimensions)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.norm_rect.Set(norm_rect);
                     node.image_dimensions.Set(dimensions);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> output,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(
                     GetCroppedRectTestInputForRotation()),
                 api3::MakePacket<NormalizedRect>(
                     GetCroppedRectWith90degreeRotation()),
                 api3::MakePacket<std::pair<int, int>>(2, 1)));
  ASSERT_TRUE(output);
  const auto& result = output.GetOrDie();
  ASSERT_EQ(result.landmark_size(), 1);
  EXPECT_NEAR(result.landmark(0).x(), 0.25, kAbsError);
  EXPECT_NEAR(result.landmark(0).y(), 0.5, kAbsError);
  EXPECT_NEAR(result.landmark(0).z(), 0.0, kAbsError);
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

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<std::array<float, 16>> matrix)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.projection_matrix.Set(matrix);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> result,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(std::move(landmarks)),
                 api3::MakePacket<std::array<float, 16>>(std::move(matrix))));
  ASSERT_TRUE(result);
  EXPECT_THAT(
      result.GetOrDie(),
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
  auto landmarks = GetCroppedRectTestInput();

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<std::array<float, 16>> matrix)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.projection_matrix.Set(matrix);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> result,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(std::move(landmarks)),
                 api3::MakePacket<std::array<float, 16>>(std::move(matrix))));
  ASSERT_TRUE(result);
  EXPECT_THAT(result.GetOrDie(),
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

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<std::array<float, 16>> matrix)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.projection_matrix.Set(matrix);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> result,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(std::move(landmarks)),
                 api3::MakePacket<std::array<float, 16>>(std::move(matrix))));
  ASSERT_TRUE(result);
  EXPECT_THAT(
      result.GetOrDie(),
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

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<std::array<float, 16>> matrix)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.projection_matrix.Set(matrix);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> result,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(std::move(landmarks)),
                 api3::MakePacket<std::array<float, 16>>(std::move(matrix))));
  ASSERT_TRUE(result);
  EXPECT_THAT(
      result.GetOrDie(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 11, y: 22, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithRotationMatrix) {
  NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<NormalizedLandmarkList>(R"pb(
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

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<NormalizedLandmarkList> landmarks,
                                  Stream<std::array<float, 16>> matrix)
                                   -> Stream<NormalizedLandmarkList> {
                     auto& node = graph.AddNode<LandmarkProjectionNode>();
                     node.projection_matrix.Set(matrix);
                     node.input_landmarks.Add(landmarks);
                     return node.output_landmarks.Add();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<NormalizedLandmarkList> result,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(std::move(landmarks)),
                 api3::MakePacket<std::array<float, 16>>(std::move(matrix))));
  ASSERT_TRUE(result);
  EXPECT_THAT(result.GetOrDie(),
              EqualsProto(ParseTextProtoOrDie<NormalizedLandmarkList>(R"pb(
                landmark { x: 0, y: 4, z: -0.5 }
              )pb")));
}

TEST(LandmarkProjectionCalculatorTest, HasCorrectRegistrationName) {
  EXPECT_EQ(LandmarkProjectionNode::GetRegistrationName(),
            "LandmarkProjectionCalculator");
}

}  // namespace
}  // namespace mediapipe::api3
