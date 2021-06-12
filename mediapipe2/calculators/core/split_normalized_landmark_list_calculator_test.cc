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

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

constexpr float kLocationVal = 3;

class SplitNormalizedLandmarkListCalculatorTest : public ::testing::Test {
 protected:
  void TearDown() { expected_landmarks_.reset(); }

  void PrepareNormalizedLandmarkList(int list_size) {
    // Prepare input landmark list.
    input_landmarks_ = absl::make_unique<NormalizedLandmarkList>();
    expected_landmarks_ = absl::make_unique<NormalizedLandmarkList>();
    for (int i = 0; i < list_size; ++i) {
      NormalizedLandmark* landmark = input_landmarks_->add_landmark();
      landmark->set_x(i * kLocationVal);
      landmark->set_y(i * kLocationVal);
      landmark->set_z(i * kLocationVal);
      // Save the landmarks for comparison after the graph runs.
      *expected_landmarks_->add_landmark() = *landmark;
    }
  }

  void ValidateListOutput(std::vector<Packet>& output_packets,
                          int expected_elements, int input_begin_index) {
    ASSERT_EQ(1, output_packets.size());
    const NormalizedLandmarkList& output_landmarks =
        output_packets[0].Get<NormalizedLandmarkList>();
    ASSERT_EQ(expected_elements, output_landmarks.landmark_size());

    for (int i = 0; i < expected_elements; ++i) {
      const NormalizedLandmark& expected_landmark =
          expected_landmarks_->landmark(input_begin_index + i);
      const NormalizedLandmark& result = output_landmarks.landmark(i);
      EXPECT_FLOAT_EQ(expected_landmark.x(), result.x());
      EXPECT_FLOAT_EQ(expected_landmark.y(), result.y());
      EXPECT_FLOAT_EQ(expected_landmark.z(), result.z());
    }
  }

  void ValidateCombinedListOutput(std::vector<Packet>& output_packets,
                                  int expected_elements,
                                  std::vector<int>& input_begin_indices,
                                  std::vector<int>& input_end_indices) {
    ASSERT_EQ(1, output_packets.size());
    ASSERT_EQ(input_begin_indices.size(), input_end_indices.size());
    const NormalizedLandmarkList& output_landmarks =
        output_packets[0].Get<NormalizedLandmarkList>();
    ASSERT_EQ(expected_elements, output_landmarks.landmark_size());
    const int num_ranges = input_begin_indices.size();

    int element_id = 0;
    for (int range_id = 0; range_id < num_ranges; ++range_id) {
      for (int i = input_begin_indices[range_id];
           i < input_end_indices[range_id]; ++i) {
        const NormalizedLandmark& expected_landmark =
            expected_landmarks_->landmark(i);
        const NormalizedLandmark& result =
            output_landmarks.landmark(element_id);
        EXPECT_FLOAT_EQ(expected_landmark.x(), result.x());
        EXPECT_FLOAT_EQ(expected_landmark.y(), result.y());
        EXPECT_FLOAT_EQ(expected_landmark.z(), result.z());
        element_id++;
      }
    }
  }

  void ValidateElementOutput(std::vector<Packet>& output_packets,
                             int input_begin_index) {
    ASSERT_EQ(1, output_packets.size());

    const NormalizedLandmark& output_landmark =
        output_packets[0].Get<NormalizedLandmark>();
    ASSERT_TRUE(output_landmark.IsInitialized());

    const NormalizedLandmark& expected_landmark =
        expected_landmarks_->landmark(input_begin_index);

    EXPECT_FLOAT_EQ(expected_landmark.x(), output_landmark.x());
    EXPECT_FLOAT_EQ(expected_landmark.y(), output_landmark.y());
    EXPECT_FLOAT_EQ(expected_landmark.z(), output_landmark.z());
  }

  std::unique_ptr<NormalizedLandmarkList> input_landmarks_ = nullptr;
  std::unique_ptr<NormalizedLandmarkList> expected_landmarks_ = nullptr;
  std::unique_ptr<CalculatorRunner> runner_ = nullptr;
};

TEST_F(SplitNormalizedLandmarkListCalculatorTest, SmokeTest) {
  PrepareNormalizedLandmarkList(/*list_size=*/5);
  ASSERT_NE(input_landmarks_, nullptr);

  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              output_stream: "range_1"
              output_stream: "range_2"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 1 end: 4 }
                  ranges: { begin: 4 end: 5 }
                }
              }
            }
          )pb");
  std::vector<Packet> range_0_packets;
  tool::AddVectorSink("range_0", &graph_config, &range_0_packets);
  std::vector<Packet> range_1_packets;
  tool::AddVectorSink("range_1", &graph_config, &range_1_packets);
  std::vector<Packet> range_2_packets;
  tool::AddVectorSink("range_2", &graph_config, &range_2_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "landmarks_in", Adopt(input_landmarks_.release()).At(Timestamp(0))));
  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ValidateListOutput(range_0_packets, /*expected_elements=*/1,
                     /*input_begin_index=*/0);
  ValidateListOutput(range_1_packets, /*expected_elements=*/3,
                     /*input_begin_index=*/1);
  ValidateListOutput(range_2_packets, /*expected_elements=*/1,
                     /*input_begin_index=*/4);

  // Fully close the graph at the end.
  MP_ASSERT_OK(graph.CloseInputStream("landmarks_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest, InvalidRangeTest) {
  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 0 }
                }
              }
            }
          )pb");

  // Run the graph.
  CalculatorGraph graph;
  // The graph should fail running because of an invalid range (begin == end).
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest,
       InvalidOutputStreamCountTest) {
  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              output_stream: "range_1"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                }
              }
            }
          )pb");

  // Run the graph.
  CalculatorGraph graph;
  // The graph should fail running because the number of output streams does not
  // match the number of range elements in the options.
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest,
       InvalidCombineOutputsMultipleOutputsTest) {
  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              output_stream: "range_1"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 2 end: 3 }
                  combine_outputs: true
                }
              }
            }
          )pb");

  // Run the graph.
  CalculatorGraph graph;
  // The graph should fail running because the number of output streams does not
  // match the number of range elements in the options.
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest,
       InvalidOverlappingRangesTest) {
  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 3 }
                  ranges: { begin: 1 end: 4 }
                  combine_outputs: true
                }
              }
            }
          )pb");

  // Run the graph.
  CalculatorGraph graph;
  // The graph should fail running because there are overlapping ranges.
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest, SmokeTestElementOnly) {
  PrepareNormalizedLandmarkList(/*list_size=*/5);
  ASSERT_NE(input_landmarks_, nullptr);

  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              output_stream: "range_1"
              output_stream: "range_2"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 2 end: 3 }
                  ranges: { begin: 4 end: 5 }
                  element_only: true
                }
              }
            }
          )pb");
  std::vector<Packet> range_0_packets;
  tool::AddVectorSink("range_0", &graph_config, &range_0_packets);
  std::vector<Packet> range_1_packets;
  tool::AddVectorSink("range_1", &graph_config, &range_1_packets);
  std::vector<Packet> range_2_packets;
  tool::AddVectorSink("range_2", &graph_config, &range_2_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "landmarks_in", Adopt(input_landmarks_.release()).At(Timestamp(0))));
  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ValidateElementOutput(range_0_packets,
                        /*input_begin_index=*/0);
  ValidateElementOutput(range_1_packets,
                        /*input_begin_index=*/2);
  ValidateElementOutput(range_2_packets,
                        /*input_begin_index=*/4);

  // Fully close the graph at the end.
  MP_ASSERT_OK(graph.CloseInputStream("landmarks_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest, SmokeTestCombiningOutputs) {
  PrepareNormalizedLandmarkList(/*list_size=*/5);
  ASSERT_NE(input_landmarks_, nullptr);

  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 2 end: 3 }
                  ranges: { begin: 4 end: 5 }
                  combine_outputs: true
                }
              }
            }
          )pb");
  std::vector<Packet> range_0_packets;
  tool::AddVectorSink("range_0", &graph_config, &range_0_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "landmarks_in", Adopt(input_landmarks_.release()).At(Timestamp(0))));
  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  std::vector<int> input_begin_indices = {0, 2, 4};
  std::vector<int> input_end_indices = {1, 3, 5};
  ValidateCombinedListOutput(range_0_packets, /*expected_elements=*/3,
                             input_begin_indices, input_end_indices);

  // Fully close the graph at the end.
  MP_ASSERT_OK(graph.CloseInputStream("landmarks_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(SplitNormalizedLandmarkListCalculatorTest,
       ElementOnlyDisablesVectorOutputs) {
  // Prepare a graph to use the SplitNormalizedLandmarkListCalculator.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "landmarks_in"
            node {
              calculator: "SplitNormalizedLandmarkListCalculator"
              input_stream: "landmarks_in"
              output_stream: "range_0"
              output_stream: "range_1"
              output_stream: "range_2"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 1 end: 4 }
                  ranges: { begin: 4 end: 5 }
                  element_only: true
                }
              }
            }
          )pb");

  // Run the graph.
  CalculatorGraph graph;
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

}  // namespace mediapipe
