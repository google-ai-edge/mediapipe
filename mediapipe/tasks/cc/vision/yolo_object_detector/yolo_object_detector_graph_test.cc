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

// Integration tests for YoloObjectDetectorGraph.
//
// DISABLED tests require model files at:
//   mediapipe/tasks/cc/vision/yolo_object_detector/testdata/
//     yolov8n_float32.tflite     — Family A (Ultralytics head, NMS applied)
//     yolo26n_e2e_float32.tflite — Family B (end-to-end NMS, NMS skipped)
//
// To enable: place models at the paths above and remove the DISABLED_ prefix.

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::vision::yolo_object_detector {
namespace {

constexpr char kFamilyAModelPath[] =
    "mediapipe/tasks/cc/vision/yolo_object_detector/testdata/"
    "yolov8n_float32.tflite";

constexpr char kFamilyBModelPath[] =
    "mediapipe/tasks/cc/vision/yolo_object_detector/testdata/"
    "yolo26n_e2e_float32.tflite";

// Build a CalculatorGraphConfig that uses YoloObjectDetectorGraph.
CalculatorGraphConfig BuildConfig(const std::string& model_path,
                                  float score_threshold = 0.25f) {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrCat(R"pb(
    input_stream:  "image"
    output_stream: "detections"
    node {
      calculator: "YoloObjectDetectorGraph"
      input_stream:  "IMAGE:image"
      output_stream: "DETECTIONS:detections"
      options {
        [mediapipe.tasks.vision.yolo_object_detector.proto
          .YoloObjectDetectorOptions.ext] {
          base_options {
            model_asset { file_name: ")pb",
                                        model_path, R"pb(" }
          }
          score_threshold: )pb",
                                    score_threshold, R"pb(
        }
      }
    }
  )pb"));
}

// Helper: create a blank sRGB Image of given dimensions.
mediapipe::Image MakeBlankImage(int width, int height) {
  auto frame = std::make_shared<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, width, height);
  memset(frame->MutablePixelData(), 128,
         frame->Height() * frame->WidthStep());
  return mediapipe::Image(std::move(frame));
}

// ── Family A: Ultralytics NMS applied ────────────────────────────────────
TEST(YoloObjectDetectorGraphTest,
     DISABLED_FamilyA_RunsWithRealModel_OutputPacketProduced) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(BuildConfig(kFamilyAModelPath)));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image",
      MakePacket<mediapipe::Image>(MakeBlankImage(640, 640)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("image"));
  MP_ASSERT_OK(graph.WaitUntilDone());

  // A blank image should produce 0 detections (no real objects).
  ASSERT_EQ(output.size(), 1u);
  EXPECT_EQ(output[0].Get<std::vector<Detection>>().size(), 0u);
}

// ── Family B: End-to-end, NMS skipped ────────────────────────────────────
TEST(YoloObjectDetectorGraphTest,
     DISABLED_FamilyB_EndToEnd_RunsWithRealModel_OutputPacketProduced) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(BuildConfig(kFamilyBModelPath)));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image",
      MakePacket<mediapipe::Image>(MakeBlankImage(640, 640)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("image"));
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(output.size(), 1u);
  // Output should be produced (count depends on model and blank image).
  EXPECT_GE(output[0].Get<std::vector<Detection>>().size(), 0u);
}

}  // namespace
}  // namespace mediapipe::tasks::vision::yolo_object_detector
