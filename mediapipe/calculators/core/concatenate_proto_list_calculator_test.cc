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

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT

namespace mediapipe {

constexpr float kLocationValue = 3;

NormalizedLandmarkList GenerateLandmarks(int landmarks_size,
                                         int value_multiplier) {
  NormalizedLandmarkList landmarks;
  for (int i = 0; i < landmarks_size; ++i) {
    NormalizedLandmark* landmark = landmarks.add_landmark();
    landmark->set_x(value_multiplier * kLocationValue);
    landmark->set_y(value_multiplier * kLocationValue);
    landmark->set_z(value_multiplier * kLocationValue);
  }
  return landmarks;
}

void ValidateCombinedLandmarks(
    const std::vector<NormalizedLandmarkList>& inputs,
    const NormalizedLandmarkList& result) {
  int element_id = 0;
  int expected_size = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    const NormalizedLandmarkList& landmarks_i = inputs[i];
    expected_size += landmarks_i.landmark_size();
    for (int j = 0; j < landmarks_i.landmark_size(); ++j) {
      const NormalizedLandmark& expected = landmarks_i.landmark(j);
      const NormalizedLandmark& got = result.landmark(element_id);
      EXPECT_FLOAT_EQ(expected.x(), got.x());
      EXPECT_FLOAT_EQ(expected.y(), got.y());
      EXPECT_FLOAT_EQ(expected.z(), got.z());
      ++element_id;
    }
  }
  EXPECT_EQ(expected_size, result.landmark_size());
}

void AddInputLandmarkLists(
    const std::vector<NormalizedLandmarkList>& input_landmarks_vec,
    int64_t timestamp, CalculatorRunner* runner) {
  for (int i = 0; i < input_landmarks_vec.size(); ++i) {
    runner->MutableInputs()->Index(i).packets.push_back(
        MakePacket<NormalizedLandmarkList>(input_landmarks_vec[i])
            .At(Timestamp(timestamp)));
  }
}

void AddInputClassificationLists(
    const std::vector<ClassificationList>& input_classifications_vec,
    int64_t timestamp, CalculatorRunner* runner) {
  for (int i = 0; i < input_classifications_vec.size(); ++i) {
    runner->MutableInputs()->Index(i).packets.push_back(
        MakePacket<ClassificationList>(input_classifications_vec[i])
            .At(Timestamp(timestamp)));
  }
}

TEST(ConcatenateNormalizedLandmarkListCalculatorTest, EmptyVectorInputs) {
  CalculatorRunner runner("ConcatenateNormalizedLandmarkListCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  NormalizedLandmarkList empty_list;
  std::vector<NormalizedLandmarkList> inputs = {empty_list, empty_list,
                                                empty_list};
  AddInputLandmarkLists(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(0, outputs[0].Get<NormalizedLandmarkList>().landmark_size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
}

TEST(ConcatenateNormalizedLandmarkListCalculatorTest, OneTimestamp) {
  CalculatorRunner runner("ConcatenateNormalizedLandmarkListCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  NormalizedLandmarkList input_0 =
      GenerateLandmarks(/*landmarks_size=*/3, /*value_multiplier=*/0);
  NormalizedLandmarkList input_1 =
      GenerateLandmarks(/*landmarks_size=*/1, /*value_multiplier=*/1);
  NormalizedLandmarkList input_2 =
      GenerateLandmarks(/*landmarks_size=*/2, /*value_multiplier=*/2);
  std::vector<NormalizedLandmarkList> inputs = {input_0, input_1, input_2};
  AddInputLandmarkLists(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  const NormalizedLandmarkList& result =
      outputs[0].Get<NormalizedLandmarkList>();
  ValidateCombinedLandmarks(inputs, result);
}

TEST(ConcatenateNormalizedLandmarkListCalculatorTest,
     TwoInputsAtTwoTimestamps) {
  CalculatorRunner runner("ConcatenateNormalizedLandmarkListCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  NormalizedLandmarkList input_0 =
      GenerateLandmarks(/*landmarks_size=*/3, /*value_multiplier=*/0);
  NormalizedLandmarkList input_1 =
      GenerateLandmarks(/*landmarks_size=*/1, /*value_multiplier=*/1);
  NormalizedLandmarkList input_2 =
      GenerateLandmarks(/*landmarks_size=*/2, /*value_multiplier=*/2);
  std::vector<NormalizedLandmarkList> inputs = {input_0, input_1, input_2};
  {
    AddInputLandmarkLists(inputs, /*timestamp=*/1, &runner);
  }
  {
    AddInputLandmarkLists(inputs, /*timestamp=*/2, &runner);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(2, outputs.size());
  {
    EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
    const NormalizedLandmarkList& result =
        outputs[0].Get<NormalizedLandmarkList>();
    ValidateCombinedLandmarks(inputs, result);
  }
  {
    EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
    const NormalizedLandmarkList& result =
        outputs[1].Get<NormalizedLandmarkList>();
    ValidateCombinedLandmarks(inputs, result);
  }
}

TEST(ConcatenateNormalizedLandmarkListCalculatorTest,
     OneEmptyStreamStillOutput) {
  CalculatorRunner runner("ConcatenateNormalizedLandmarkListCalculator",
                          /*options_string=*/"", /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  NormalizedLandmarkList input_0 =
      GenerateLandmarks(/*landmarks_size=*/3, /*value_multiplier=*/0);
  std::vector<NormalizedLandmarkList> inputs = {input_0};
  AddInputLandmarkLists(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  const NormalizedLandmarkList& result =
      outputs[0].Get<NormalizedLandmarkList>();
  ValidateCombinedLandmarks(inputs, result);
}

TEST(ConcatenateNormalizedLandmarkListCalculatorTest, OneEmptyStreamNoOutput) {
  CalculatorRunner runner("ConcatenateNormalizedLandmarkListCalculator",
                          /*options_string=*/
                          "[mediapipe.ConcatenateVectorCalculatorOptions.ext]: "
                          "{only_emit_if_all_present: true}",
                          /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  NormalizedLandmarkList input_0 =
      GenerateLandmarks(/*landmarks_size=*/3, /*value_multiplier=*/0);
  std::vector<NormalizedLandmarkList> inputs = {input_0};
  AddInputLandmarkLists(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(0, outputs.size());
}

TEST(ConcatenateClassificationListCalculatorTest, OneTimestamp) {
  CalculatorRunner runner("ConcatenateClassificationListCalculator",
                          /*options_string=*/
                          "[mediapipe.ConcatenateVectorCalculatorOptions.ext]: "
                          "{only_emit_if_all_present: true}",
                          /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  auto input_0 = ParseTextProtoOrDie<ClassificationList>(R"pb(
    classification: { index: 0 score: 0.2 label: "test_0" }
    classification: { index: 1 score: 0.3 label: "test_1" }
    classification: { index: 2 score: 0.4 label: "test_2" }
  )pb");
  auto input_1 = ParseTextProtoOrDie<ClassificationList>(R"pb(
    classification: { index: 3 score: 0.2 label: "test_3" }
    classification: { index: 4 score: 0.3 label: "test_4" }
  )pb");
  std::vector<ClassificationList> inputs = {input_0, input_1};
  AddInputClassificationLists(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  auto result = outputs[0].Get<ClassificationList>();
  EXPECT_THAT(ParseTextProtoOrDie<ClassificationList>(R"pb(
                classification: { index: 0 score: 0.2 label: "test_0" }
                classification: { index: 1 score: 0.3 label: "test_1" }
                classification: { index: 2 score: 0.4 label: "test_2" }
                classification: { index: 3 score: 0.2 label: "test_3" }
                classification: { index: 4 score: 0.3 label: "test_4" }
              )pb"),
              EqualsProto(result));
}

}  // namespace mediapipe
