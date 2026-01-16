// Copyright 2023 The MediaPipe Authors.
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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace api2 {
namespace {

using Node = ::mediapipe::CalculatorGraphConfig::Node;

Landmark CreateLandmark(float x, float y, float z) {
  Landmark lmk;
  lmk.set_x(x);
  lmk.set_y(y);
  lmk.set_z(z);
  return lmk;
}

Landmark CreateLandmark(float x, float y, float z, float visibility,
                        float presence) {
  Landmark lmk;
  lmk.set_x(x);
  lmk.set_y(y);
  lmk.set_z(z);
  lmk.set_visibility(visibility);
  lmk.set_presence(presence);
  return lmk;
}

struct LandmarksTransformationestCase {
  std::string test_name;
  std::string transformations;
  std::vector<Landmark> in_landmarks;
  std::vector<Landmark> out_landmarks;
};

using LandmarksTransformationest =
    ::testing::TestWithParam<LandmarksTransformationestCase>;

TEST_P(LandmarksTransformationest, LandmarksTransformationest) {
  const LandmarksTransformationestCase& tc = GetParam();

  // Prepare graph.
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(absl::Substitute(
      R"(
      calculator: "LandmarksTransformationCalculator"
      input_stream: "LANDMARKS:in_landmarks"
      output_stream: "LANDMARKS:out_landmarks"
      options: {
        [mediapipe.LandmarksTransformationCalculatorOptions.ext] {
          $0
        }
      }
  )",
      tc.transformations)));

  // In landmarks.
  LandmarkList in_landmarks;
  for (auto& lmk : tc.in_landmarks) {
    *in_landmarks.add_landmark() = lmk;
  }

  // Send landmarks to the graph.
  runner.MutableInputs()
      ->Tag("LANDMARKS")
      .packets.push_back(MakePacket<LandmarkList>(std::move(in_landmarks))
                             .At(mediapipe::Timestamp(0)));

  // Run the graph.
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Tag("LANDMARKS").packets;
  EXPECT_EQ(1, output_packets.size());

  const auto& out_landmarks = output_packets[0].Get<LandmarkList>();
  EXPECT_EQ(out_landmarks.landmark_size(), tc.out_landmarks.size());
  for (int i = 0; i < out_landmarks.landmark_size(); ++i) {
    auto& lmk = out_landmarks.landmark(i);
    auto& exp_lmk = tc.out_landmarks[i];
    EXPECT_EQ(lmk.x(), exp_lmk.x()) << "Unexpected lmk[" << i << "].x";
    EXPECT_EQ(lmk.y(), exp_lmk.y()) << "Unexpected lmk[" << i << "].y";
    EXPECT_EQ(lmk.z(), exp_lmk.z()) << "Unexpected lmk[" << i << "].z";
    if (exp_lmk.has_visibility()) {
      EXPECT_EQ(lmk.visibility(), exp_lmk.visibility())
          << "Unexpected lmk[" << i << "].visibility";
    }
    if (exp_lmk.has_presence()) {
      EXPECT_EQ(lmk.presence(), exp_lmk.presence())
          << "Unexpected lmk[" << i << "].presence";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LandmarksTransformationests, LandmarksTransformationest,
    testing::ValuesIn<LandmarksTransformationestCase>({
        {"NoTransformations",
         "",
         {CreateLandmark(1, 2, 3), CreateLandmark(4, 5, 6)},
         {CreateLandmark(1, 2, 3), CreateLandmark(4, 5, 6)}},

        {"NormalizeTranslation_OneLandmark",
         "transformation: { normalize_translation: {} }",
         {CreateLandmark(2, 2, 2)},
         {CreateLandmark(0, 0, 0)}},
        {"NormalizeTranslation_TwoLandmarks",
         "transformation: { normalize_translation: {} }",
         {CreateLandmark(2, 2, 2), CreateLandmark(4, 4, 4)},
         {CreateLandmark(-1, -1, -1), CreateLandmark(1, 1, 1)}},
        {"NormalizeTranslation_ThreeLandmarks",
         "transformation: { normalize_translation: {} }",
         {CreateLandmark(2, 2, 2), CreateLandmark(4, 4, 4),
          CreateLandmark(9, 9, 9)},
         {CreateLandmark(-3, -3, -3), CreateLandmark(-1, -1, -1),
          CreateLandmark(4, 4, 4)}},
        {"NormalizeTranslation_VisibilityAndPresence",
         "transformation: { normalize_translation: {} }",
         {CreateLandmark(0, 0, 0, 4, 5)},
         {CreateLandmark(0, 0, 0, 4, 5)}},

        {"FlipAxis_X",
         "transformation: { flip_axis: { flip_x: true } }",
         {CreateLandmark(2, 2, 2)},
         {CreateLandmark(-2, 2, 2)}},
        {"FlipAxis_Y",
         "transformation: { flip_axis: { flip_y: true } }",
         {CreateLandmark(2, 2, 2)},
         {CreateLandmark(2, -2, 2)}},
        {"FlipAxis_Z",
         "transformation: { flip_axis: { flip_z: true } }",
         {CreateLandmark(2, 2, 2)},
         {CreateLandmark(2, 2, -2)}},
        {"FlipAxis_VisibilityAndPresence",
         "transformation: { flip_axis: { flip_x: true } }",
         {CreateLandmark(0, 0, 0, 4, 5)},
         {CreateLandmark(0, 0, 0, 4, 5)}},
    }),
    [](const testing::TestParamInfo<LandmarksTransformationest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace api2
}  // namespace mediapipe
