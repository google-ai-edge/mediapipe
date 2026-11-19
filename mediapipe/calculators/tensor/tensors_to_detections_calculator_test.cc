// Copyright 2026 The MediaPipe Authors.
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

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"        // from @com_google_absl
#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Matcher;
using ::testing::Property;
using ::testing::ResultOf;
using ::testing::TestWithParam;
using ::testing::Values;

// API3 contract of TensorsToDetectionsCalculator, limited to what the tests
// use.
struct TensorsToDetectionsNode : api3::Node<"TensorsToDetectionsCalculator"> {
  template <typename S>
  struct Contract {
    api3::Input<S, std::vector<Tensor>> tensors{"TENSORS"};
    api3::Output<S, std::vector<Detection>> detections{"DETECTIONS"};
    api3::Options<S, mediapipe::TensorsToDetectionsCalculatorOptions> options;
  };
};

// Builds a graph decoding the outputs of a model with in-model
// post-processing, which are 4 tensors: scores [1, N], boxes [1, N, 4] (xmin,
// ymin, xmax, ymax), the number of detections [1] and classes [1, N]. Both the
// tensor order and the box coordinate order differ from the calculator
// defaults.
api3::Stream<std::vector<Detection>> BuildInModelNmsGraph(
    api3::GenericGraph& graph, api3::Stream<std::vector<Tensor>> tensors) {
  auto& node = graph.AddNode<TensorsToDetectionsNode>();
  node.tensors.Set(tensors);
  auto& options = *node.options.Mutable();
  options.set_num_classes(3);
  options.set_num_coords(4);
  options.set_min_score_thresh(0.5f);
  auto& tensor_mapping = *options.mutable_tensor_mapping();
  tensor_mapping.set_scores_tensor_index(0);
  tensor_mapping.set_detections_tensor_index(1);
  tensor_mapping.set_num_detections_tensor_index(2);
  tensor_mapping.set_classes_tensor_index(3);
  auto& box_indices = *options.mutable_box_boundaries_indices();
  box_indices.set_xmin(0);
  box_indices.set_ymin(1);
  box_indices.set_xmax(2);
  box_indices.set_ymax(3);
  return node.detections.Get();
}

struct TensorData {
  Tensor::Shape shape;
  std::vector<float> values;
};

// Model outputs for BuildInModelNmsGraph() with room for two detections: class
// 1 at score 0.9 and class 2 at score 0.3 (filtered out by min_score_thresh).
// `num_detections` is the value of the number-of-detections tensor.
std::vector<TensorData> InModelNmsOutputs(float num_detections = 2.0f) {
  return {
      // Scores.
      {.shape = {1, 2}, .values = {0.9f, 0.3f}},
      // Boxes (xmin, ymin, xmax, ymax).
      {.shape = {1, 2, 4},
       .values = {0.2f, 0.1f, 0.6f, 0.5f,  //
                  0.3f, 0.3f, 0.4f, 0.4f}},
      // Number of detections.
      {.shape = {1}, .values = {num_detections}},
      // Classes.
      {.shape = {1, 2}, .values = {1.0f, 2.0f}},
  };
}

// Returns InModelNmsOutputs(num_detections) as CPU tensors.
std::vector<Tensor> MakeInModelNmsCpuTensors(float num_detections = 2.0f) {
  std::vector<Tensor> tensors;
  for (const TensorData& data : InModelNmsOutputs(num_detections)) {
    ABSL_CHECK_EQ(data.values.size(), data.shape.num_elements());
    Tensor& tensor =
        tensors.emplace_back(Tensor::ElementType::kFloat32, data.shape);
    Tensor::CpuWriteView view = tensor.GetCpuWriteView();
    std::copy(data.values.begin(), data.values.end(), view.buffer<float>());
  }
  return tensors;
}

// Decodes `tensors` with `runner`, created for BuildInModelNmsGraph().
template <typename RunnerT>
absl::StatusOr<std::vector<Detection>> Decode(RunnerT& runner,
                                              std::vector<Tensor> tensors) {
  ABSL_ASSIGN_OR_RETURN(
      api3::Packet<std::vector<Detection>> detections,
      runner.Run(api3::MakePacket<std::vector<Tensor>>(std::move(tensors))));
  if (!detections) {
    return absl::InternalError("Expected an output packet.");
  }
  return detections.GetOrDie();
}

// Matches the detections decoded from InModelNmsOutputs(): only the class 1
// detection (score 0.9) passes min_score_thresh.
Matcher<std::vector<Detection>> IsSingleClass1Detection() {
  using Box = LocationData::RelativeBoundingBox;
  return ElementsAre(AllOf(
      ResultOf([](const Detection& d) { return d.label_id(); }, ElementsAre(1)),
      ResultOf([](const Detection& d) { return d.score(); },
               ElementsAre(FloatNear(0.9f, 1e-6f))),
      ResultOf(
          [](const Detection& d) {
            return d.location_data().relative_bounding_box();
          },
          AllOf(Property(&Box::xmin, FloatNear(0.2f, 1e-6f)),
                Property(&Box::ymin, FloatNear(0.1f, 1e-6f)),
                Property(&Box::width, FloatNear(0.4f, 1e-6f)),
                Property(&Box::height, FloatNear(0.4f, 1e-6f))))));
}

TEST(TensorsToDetectionsCalculatorTest, DecodesInModelNmsCpuTensors) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          api3::Runner::For(BuildInModelNmsGraph).Create());

  EXPECT_THAT(Decode(runner, MakeInModelNmsCpuTensors()),
              IsOkAndHolds(IsSingleClass1Detection()));
}

// The parameter is an invalid value of the number-of-detections tensor.
class InvalidNumDetectionsTest : public TestWithParam<float> {};

TEST_P(InvalidNumDetectionsTest, IsRejected) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          api3::Runner::For(BuildInModelNmsGraph).Create());

  EXPECT_THAT(Decode(runner, MakeInModelNmsCpuTensors(GetParam())),
              StatusIs(absl::StatusCode::kInternal));
}

// InModelNmsOutputs() has room for 2 detections.
INSTANTIATE_TEST_SUITE_P(TensorsToDetectionsCalculatorTest,
                         InvalidNumDetectionsTest,
                         Values(-1.0f, 3.0f,
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity()));

}  // namespace
}  // namespace mediapipe
