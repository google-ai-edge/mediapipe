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
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/graph_test_base.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/options_map.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"
#include "mediapipe/framework/tool/test_util.h"

namespace mediapipe {
namespace api2 {
namespace {

using mediapipe::Detection;
using mediapipe::InferenceCalculatorOptions_Delegate;
using testing::ElementsAre;
using testing::EqualsProto;
using testing::proto::Approximately;

struct Param {
  std::string name;          // Appended to the test name.
  std::string impl_suffix;   // Expected InferenceCalculator backend.
  std::string golden_image;  // Expected golden image
  InferenceCalculatorOptions_Delegate delegate;
};

const std::vector<Param>& GetParams() {
  static auto all_params = [] {
    static std::vector<Param> p;
    p.push_back({"TfLite", "Cpu", "face_detection_expected.png"});
    p.back().delegate.mutable_tflite();
#if TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR
    // Metal is not available on the iOS simulator.
    p.push_back({"Metal", "Metal", "face_detection_expected.png"});
    p.back().delegate.mutable_gpu();
#endif                // TARGET_IPHONE_SIMULATOR
#if __ANDROID__ && 0  // Disabled for now since emulator can't go GLESv3
    p.push_back({"Gl", "Gl", "face_detection_expected.png"});
    p.back().delegate.mutable_gpu();
    // This requires API level 27
    p.push_back({"NnApi", "Cpu", "face_detection_expected.png"});
    p.back().delegate.mutable_nnapi();
#endif  // __ANDROID__
#if !defined(__ANDROID__) && !defined(__EMSCRIPTEN__) && \
    !defined(TARGET_OS_IPHONE)  // Linux tests
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    p.push_back({"GlAdvanced", "GlAdvanced", "face_detection_expected_gl.png"});
#else
    p.push_back({"GlAdvanced", "Cpu", "face_detection_expected.png"});
#endif
    p.back().delegate.mutable_gpu()->set_use_advanced_gpu_api(true);
    p.back().delegate.mutable_gpu()->set_api(
        InferenceCalculatorOptions_Delegate::Gpu::OPENGL);
#endif  // !defined(__ANDROID__) && !defined(__EMSCRIPTEN__) &&
        // !defined(TARGET_OS_IPHONE)
    p.push_back({"XnnPack", "Cpu", "face_detection_expected.png"});
    p.back().delegate.mutable_xnnpack();
    return p;
  }();
  return all_params;
}

class InferenceCalculatorTest : public testing::TestWithParam<Param> {
 protected:
  void SetDelegateForParam(mediapipe::CalculatorGraphConfig_Node* node) {
    auto options_map = tool::MutableOptionsMap().Initialize(*node);
    auto options = options_map.Get<mediapipe::InferenceCalculatorOptions>();
    *options.mutable_delegate() = GetParam().delegate;
    options_map.Set(options);
  }
};

TEST_P(InferenceCalculatorTest, TestBackendSelection) {
  CalculatorGraphConfig config;
  auto node = config.add_node();
  node->set_calculator("InferenceCalculator");
  SetDelegateForParam(node);
  MP_ASSERT_OK(tool::ExpandSubgraphs(&config));
  EXPECT_EQ(config.node(0).calculator(),
            absl::StrCat("InferenceCalculator", GetParam().impl_suffix));
}

TEST_P(InferenceCalculatorTest, TestFaceDetection) {
  CalculatorGraphConfig config;
  ASSERT_TRUE(LoadTestGraph(
      &config, file::JoinPath(GetTestRootDir(),
                              "mediapipe/calculators/tensor/"
                              "testdata/face_detection_test.binarypb")));

  // Expand subgraphs to find any nested instances of InferenceCalculator.
  MP_ASSERT_OK(tool::ExpandSubgraphs(&config));
  int found = 0;
  for (auto& node : *config.mutable_node()) {
    // The InferenceCalculator subgraph itself will have expanded to a specific
    // implementation. Replace it.
    // TODO: make it possible to exclude it from expansion above.
    if (absl::StartsWith(node.calculator(), "InferenceCalculator")) {
      ++found;
      node.set_calculator("InferenceCalculator");
      SetDelegateForParam(&node);
    }
  }
  ASSERT_EQ(found, 1);

  std::vector<mediapipe::Packet> detection_packets;
  tool::AddVectorSink("detections", &config, &detection_packets);
  std::vector<mediapipe::Packet> rendering_packets;
  tool::AddVectorSink("rendering", &config, &rendering_packets);

  // Load test image.
  std::unique_ptr<ImageFrame> input_image = LoadTestPng(
      file::JoinPath(GetTestRootDir(), "mediapipe/objc/testdata/sergey.png"));
  ASSERT_THAT(input_image, testing::NotNull());

  std::unique_ptr<ImageFrame> expected_image =
      LoadTestPng(file::JoinPath(GetTestRootDir(),
                                 "mediapipe/calculators/tensor/"
                                 "testdata",
                                 GetParam().golden_image));
  ASSERT_THAT(expected_image, testing::NotNull());

  std::string binary;
  Detection expected_detection;
  MP_ASSERT_OK(
      file::GetContents(file::JoinPath(GetTestRootDir(),
                                       "mediapipe/calculators/tensor/"
                                       "testdata/expected_detection.binarypb"),
                        &binary));
  expected_detection.ParseFromArray(binary.data(), binary.size());

  // Prepare test inputs.
  std::unordered_map<std::string, std::unique_ptr<ImageFrame>> input_streams;
  input_streams.insert(std::make_pair("image", std::move(input_image)));
  std::string output_stream = "rendering";

  // Test graph with relaxed color difference tolerance.
  // Compare with CPU generated image.
  Timestamp ts0 = Timestamp(0);
  TestGraphConfig(config, input_streams, output_stream, expected_image, {}, ts0,
                  2.0, 2.0, 1.0);

  ASSERT_EQ(detection_packets.size(), 1);
  std::vector<Detection> dets =
      detection_packets[0].Get<std::vector<Detection>>();
#if !defined(MEDIAPIPE_PROTO_LITE)
  // Approximately is not available with lite protos (b/178137094).
  constexpr float kEpison = 0.001;
  EXPECT_THAT(dets, ElementsAre(Approximately(EqualsProto(expected_detection),
                                              kEpison)));
#endif
}

INSTANTIATE_TEST_SUITE_P(Implementation, InferenceCalculatorTest,
                         testing::ValuesIn(GetParams()),
                         [](const testing::TestParamInfo<Param>& info) {
                           return info.param.name;
                         });

}  // namespace
}  // namespace api2
}  // namespace mediapipe
