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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";
constexpr char kLandmarksTag[] = "LANDMARKS";

NormalizedLandmark CreateLandmark(float x, float y) {
  NormalizedLandmark landmark;
  landmark.set_x(x);
  landmark.set_y(y);
  return landmark;
}

CalculatorGraphConfig::Node GetDefaultNode() {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "LandmarkLetterboxRemovalCalculator"
    input_stream: "LANDMARKS:landmarks"
    input_stream: "LETTERBOX_PADDING:letterbox_padding"
    output_stream: "LANDMARKS:adjusted_landmarks"
  )pb");
}

TEST(LandmarkLetterboxRemovalCalculatorTest, PaddingLeftRight) {
  CalculatorRunner runner(GetDefaultNode());

  auto landmarks = absl::make_unique<NormalizedLandmarkList>();
  *landmarks->add_landmark() = CreateLandmark(0.5f, 0.5f);
  *landmarks->add_landmark() = CreateLandmark(0.2f, 0.2f);
  *landmarks->add_landmark() = CreateLandmark(0.7f, 0.7f);
  runner.MutableInputs()
      ->Tag(kLandmarksTag)
      .packets.push_back(
          Adopt(landmarks.release()).At(Timestamp::PostStream()));

  auto padding = absl::make_unique<std::array<float, 4>>(
      std::array<float, 4>{0.2f, 0.f, 0.3f, 0.f});
  runner.MutableInputs()
      ->Tag(kLetterboxPaddingTag)
      .packets.push_back(Adopt(padding.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kLandmarksTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& output_landmarks = output[0].Get<NormalizedLandmarkList>();

  EXPECT_EQ(output_landmarks.landmark_size(), 3);

  EXPECT_THAT(output_landmarks.landmark(0).x(), testing::FloatNear(0.6f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(0).y(), testing::FloatNear(0.5f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(1).x(), testing::FloatNear(0.0f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(1).y(), testing::FloatNear(0.2f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(2).x(), testing::FloatNear(1.0f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(2).y(), testing::FloatNear(0.7f, 1e-5));
}

TEST(LandmarkLetterboxRemovalCalculatorTest, PaddingTopBottom) {
  CalculatorRunner runner(GetDefaultNode());

  auto landmarks = absl::make_unique<NormalizedLandmarkList>();
  NormalizedLandmark* landmark = landmarks->add_landmark();
  *landmark = CreateLandmark(0.5f, 0.5f);
  landmark = landmarks->add_landmark();
  *landmark = CreateLandmark(0.2f, 0.2f);
  landmark = landmarks->add_landmark();
  *landmark = CreateLandmark(0.7f, 0.7f);
  runner.MutableInputs()
      ->Tag(kLandmarksTag)
      .packets.push_back(
          Adopt(landmarks.release()).At(Timestamp::PostStream()));

  auto padding = absl::make_unique<std::array<float, 4>>(
      std::array<float, 4>{0.0f, 0.2f, 0.0f, 0.3f});
  runner.MutableInputs()
      ->Tag(kLetterboxPaddingTag)
      .packets.push_back(Adopt(padding.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kLandmarksTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& output_landmarks = output[0].Get<NormalizedLandmarkList>();

  EXPECT_EQ(output_landmarks.landmark_size(), 3);

  EXPECT_THAT(output_landmarks.landmark(0).x(), testing::FloatNear(0.5f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(0).y(), testing::FloatNear(0.6f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(1).x(), testing::FloatNear(0.2f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(1).y(), testing::FloatNear(0.0f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(2).x(), testing::FloatNear(0.7f, 1e-5));
  EXPECT_THAT(output_landmarks.landmark(2).y(), testing::FloatNear(1.0f, 1e-5));
}

}  // namespace mediapipe
