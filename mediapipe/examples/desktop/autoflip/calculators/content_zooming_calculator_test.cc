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

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/content_zooming_calculator.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/content_zooming_calculator_state.h"
#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {
namespace {

constexpr char kFirstCropRectTag[] = "FIRST_CROP_RECT";
constexpr char kStateCacheTag[] = "STATE_CACHE";
constexpr char kCropRectTag[] = "CROP_RECT";
constexpr char kBordersTag[] = "BORDERS";
constexpr char kSalientRegionsTag[] = "SALIENT_REGIONS";
constexpr char kVideoTag[] = "VIDEO";
constexpr char kMaxZoomFactorPctTag[] = "MAX_ZOOM_FACTOR_PCT";
constexpr char kAnimateZoomTag[] = "ANIMATE_ZOOM";
constexpr char kVideoSizeTag[] = "VIDEO_SIZE";
constexpr char kDetectionsTag[] = "DETECTIONS";

const char kConfigA[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO:camera_frames"
    input_stream: "SALIENT_REGIONS:detection_set"
    output_stream: "BORDERS:borders"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        max_zoom_value_deg: 0
        kinematic_options_zoom {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_tilt {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_pan {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
      }
    }
    )";

const char kConfigB[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO:camera_frames"
    input_stream: "SALIENT_REGIONS:detection_set"
    output_stream: "BORDERS:borders"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        target_size {
          width: 1000
          height: 500
        }
        max_zoom_value_deg: 0
        kinematic_options_zoom {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_tilt {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_pan {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
      }
    }
    )";

const char kConfigC[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO_SIZE:size"
    input_stream: "SALIENT_REGIONS:detection_set"
    output_stream: "BORDERS:borders"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        max_zoom_value_deg: 0
        kinematic_options_zoom {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_tilt {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_pan {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
      }
    }
    )";

const char kConfigD[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO_SIZE:size"
    input_stream: "DETECTIONS:detections"
    output_stream: "CROP_RECT:rect"
    output_stream: "FIRST_CROP_RECT:first_rect"
    output_stream: "NORMALIZED_CROP_RECT:float_rect"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        max_zoom_value_deg: 0
        kinematic_options_zoom {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_tilt {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_pan {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
      }
    }
    )";

const char kConfigE[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO_SIZE:size"
    input_stream: "DETECTIONS:detections"
    input_stream: "ANIMATE_ZOOM:animate_zoom"
    output_stream: "CROP_RECT:rect"
    output_stream: "FIRST_CROP_RECT:first_rect"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        max_zoom_value_deg: 0
        kinematic_options_zoom {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_tilt {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_pan {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
      }
    }
    )";

const char kConfigF[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO_SIZE:size"
    input_stream: "DETECTIONS:detections"
    input_stream: "MAX_ZOOM_FACTOR_PCT:max_zoom_factor_pct"
    output_stream: "CROP_RECT:rect"
    output_stream: "FIRST_CROP_RECT:first_rect"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        max_zoom_value_deg: 0
        kinematic_options_zoom {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_tilt {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
        kinematic_options_pan {
          min_motion_to_reframe: 1.2
          max_velocity: 18
        }
      }
    }
    )";

void CheckBorder(const StaticFeatures& static_features, int width, int height,
                 int top_border, int bottom_border) {
  ASSERT_EQ(2, static_features.border().size());
  auto part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), 0);
  EXPECT_EQ(part.border_position().width(), width);
  EXPECT_EQ(part.border_position().height(), top_border);
  EXPECT_EQ(Border::TOP, part.relative_position());

  part = static_features.border(1);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), height - bottom_border);
  EXPECT_EQ(part.border_position().width(), width);
  EXPECT_EQ(part.border_position().height(), bottom_border);
  EXPECT_EQ(Border::BOTTOM, part.relative_position());
}

struct AddDetectionFlags {
  std::optional<bool> animated_zoom;
  std::optional<int> max_zoom_factor_percent;
};

void AddDetectionFrameSize(const cv::Rect_<float>& position, const int64_t time,
                           const int width, const int height,
                           CalculatorRunner* runner,
                           const AddDetectionFlags& flags = {}) {
  auto detections = std::make_unique<std::vector<mediapipe::Detection>>();
  if (position.width > 0 && position.height > 0) {
    mediapipe::Detection detection;
    detection.mutable_location_data()->set_format(
        mediapipe::LocationData::RELATIVE_BOUNDING_BOX);
    detection.mutable_location_data()
        ->mutable_relative_bounding_box()
        ->set_height(position.height);
    detection.mutable_location_data()
        ->mutable_relative_bounding_box()
        ->set_width(position.width);
    detection.mutable_location_data()
        ->mutable_relative_bounding_box()
        ->set_xmin(position.x);
    detection.mutable_location_data()
        ->mutable_relative_bounding_box()
        ->set_ymin(position.y);
    detections->push_back(detection);
  }
  runner->MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(Adopt(detections.release()).At(Timestamp(time)));

  auto input_size = ::absl::make_unique<std::pair<int, int>>(width, height);
  runner->MutableInputs()
      ->Tag(kVideoSizeTag)
      .packets.push_back(Adopt(input_size.release()).At(Timestamp(time)));

  if (flags.animated_zoom.has_value()) {
    runner->MutableInputs()
        ->Tag(kAnimateZoomTag)
        .packets.push_back(
            mediapipe::MakePacket<bool>(flags.animated_zoom.value())
                .At(Timestamp(time)));
  }

  if (flags.max_zoom_factor_percent.has_value()) {
    runner->MutableInputs()
        ->Tag(kMaxZoomFactorPctTag)
        .packets.push_back(
            mediapipe::MakePacket<int>(flags.max_zoom_factor_percent.value())
                .At(Timestamp(time)));
  }
}

void AddDetection(const cv::Rect_<float>& position, const int64_t time,
                  CalculatorRunner* runner) {
  AddDetectionFrameSize(position, time, 1000, 1000, runner);
}

void CheckCropRectFloats(const float x_center, const float y_center,
                         const float width, const float height,
                         const int frame_number,
                         const CalculatorRunner::StreamContentsSet& output) {
  ASSERT_GT(output.Tag("NORMALIZED_CROP_RECT").packets.size(), frame_number);
  auto float_rect = output.Tag("NORMALIZED_CROP_RECT")
                        .packets[frame_number]
                        .Get<mediapipe::NormalizedRect>();

  EXPECT_FLOAT_EQ(float_rect.x_center(), x_center);
  EXPECT_FLOAT_EQ(float_rect.y_center(), y_center);
  EXPECT_FLOAT_EQ(float_rect.width(), width);
  EXPECT_FLOAT_EQ(float_rect.height(), height);
}

void CheckCropRect(const int x_center, const int y_center, const int width,
                   const int height, const int frame_number,
                   const std::vector<Packet>& output_packets) {
  ASSERT_GT(output_packets.size(), frame_number);
  const auto& rect = output_packets[frame_number].Get<mediapipe::Rect>();
  EXPECT_EQ(rect.x_center(), x_center);
  EXPECT_EQ(rect.y_center(), y_center);
  EXPECT_EQ(rect.width(), width);
  EXPECT_EQ(rect.height(), height);
}
TEST(ContentZoomingCalculatorTest, ZoomTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(.1);
  location->set_width(.1);
  location->set_x(.4);
  location->set_y(.5);

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag(kSalientRegionsTag)
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  CheckBorder(static_features, 1000, 1000, 494, 394);
}

TEST(ContentZoomingCalculatorTest, ZoomTestFullPTZ) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD));
  AddDetection(cv::Rect_<float>(.4, .5, .1, .1), 0, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(450, 550, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, PanConfig) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_min_motion_to_reframe(0.0);
  options->mutable_kinematic_options_pan()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_tilt()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_zoom()->set_min_motion_to_reframe(50.0);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.4, .5, .1, .1), 0, runner.get());
  AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 1000000, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(450, 550, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(483, 550, 111, 111, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, PanConfigWithCache) {
  mediapipe::autoflip::ContentZoomingCalculatorStateCacheType cache;
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  config.add_input_side_packet("STATE_CACHE:state_cache");
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_min_motion_to_reframe(0.0);
  options->mutable_kinematic_options_pan()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_tilt()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_zoom()->set_min_motion_to_reframe(50.0);
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetection(cv::Rect_<float>(.4, .5, .1, .1), 0, runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(450, 550, 111, 111, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 1000000, runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(483, 550, 111, 111, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
  // Now repeat the last frame for a new runner without the cache to see a reset
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(nullptr);
    AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 2000000, runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(525, 625, 166, 166, 0,  // Without a cache, state was lost.
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
}

TEST(ContentZoomingCalculatorTest, TiltConfig) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_tilt()->set_min_motion_to_reframe(0.0);
  options->mutable_kinematic_options_tilt()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_zoom()->set_min_motion_to_reframe(50.0);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.4, .5, .1, .1), 0, runner.get());
  AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 1000000, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(450, 550, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(450, 583, 111, 111, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ZoomConfig) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_tilt()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_zoom()->set_min_motion_to_reframe(0.0);
  options->mutable_kinematic_options_zoom()->set_update_rate_seconds(2);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.4, .5, .1, .1), 0, runner.get());
  AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 1000000, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(450, 550, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(450, 550, 138, 138, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ZoomConfigWithCache) {
  mediapipe::autoflip::ContentZoomingCalculatorStateCacheType cache;
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  config.add_input_side_packet("STATE_CACHE:state_cache");
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_tilt()->set_min_motion_to_reframe(50.0);
  options->mutable_kinematic_options_zoom()->set_min_motion_to_reframe(0.0);
  options->mutable_kinematic_options_zoom()->set_update_rate_seconds(2);
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetection(cv::Rect_<float>(.4, .5, .1, .1), 0, runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(450, 550, 111, 111, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 1000000, runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(450, 550, 138, 138, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
  // Now repeat the last frame for a new runner without the cache to see a reset
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(nullptr);
    AddDetection(cv::Rect_<float>(.45, .55, .15, .15), 2000000, runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(525, 625, 166, 166, 0,  // Without a cache, state was lost.
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
}

TEST(ContentZoomingCalculatorTest, MinAspectBorderValues) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigB));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(1);
  location->set_width(1);
  location->set_x(0);
  location->set_y(0);

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag(kSalientRegionsTag)
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  CheckBorder(static_features, 1000, 1000, 250, 250);
}

TEST(ContentZoomingCalculatorTest, TwoFacesWide) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(.2);
  location->set_width(.2);
  location->set_x(.2);
  location->set_y(.4);

  location = detection->mutable_location_normalized();
  location->set_height(.2);
  location->set_width(.2);
  location->set_x(.6);
  location->set_y(.4);

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag(kSalientRegionsTag)
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();

  CheckBorder(static_features, 1000, 1000, 389, 389);
}

TEST(ContentZoomingCalculatorTest, NoDetectionOnInit) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));
  auto detection_set = std::make_unique<DetectionSet>();

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag(kSalientRegionsTag)
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();

  CheckBorder(static_features, 1000, 1000, 0, 0);
}

TEST(ContentZoomingCalculatorTest, ZoomTestPairSize) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigC));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(.1);
  location->set_width(.1);
  location->set_x(.4);
  location->set_y(.5);

  auto input_size = ::absl::make_unique<std::pair<int, int>>(1000, 1000);
  runner->MutableInputs()
      ->Tag(kVideoSizeTag)
      .packets.push_back(Adopt(input_size.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag(kSalientRegionsTag)
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  CheckBorder(static_features, 1000, 1000, 494, 394);
}

TEST(ContentZoomingCalculatorTest, ZoomTestNearOutsideBorder) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_tilt()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_zoom()->set_update_rate_seconds(2);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.95, .95, .05, .05), 0, runner.get());
  AddDetection(cv::Rect_<float>(.9, .9, .1, .1), 1000000, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(972, 972, 55, 55, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(944, 944, 83, 83, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ZoomTestNearInsideBorder) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->mutable_kinematic_options_pan()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_tilt()->set_update_rate_seconds(2);
  options->mutable_kinematic_options_zoom()->set_update_rate_seconds(2);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(0, 0, .05, .05), 0, runner.get());
  AddDetection(cv::Rect_<float>(0, 0, .1, .1), 1000000, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(28, 28, 55, 55, 0, runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(56, 56, 83, 83, 1, runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, VerticalShift) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_detection_shift_vertical(0.2);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.1, .1, .1, .1), 0, runner.get());
  MP_ASSERT_OK(runner->Run());
  // 1000px * .1 offset + 1000*.1*.1 shift = 170
  CheckCropRect(150, 170, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRectFloats(150 / 1000.0, 170 / 1000.0, 111 / 1000.0, 111 / 1000.0, 0,
                      runner->Outputs());
}

TEST(ContentZoomingCalculatorTest, HorizontalShift) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_detection_shift_horizontal(0.2);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.1, .1, .1, .1), 0, runner.get());
  MP_ASSERT_OK(runner->Run());
  // 1000px * .1 offset + 1000*.1*.1 shift = 170
  CheckCropRect(170, 150, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRectFloats(170 / 1000.0, 150 / 1000.0, 111 / 1000.0, 111 / 1000.0, 0,
                      runner->Outputs());
}

TEST(ContentZoomingCalculatorTest, ShiftOutsideBounds) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_detection_shift_vertical(-0.2);
  options->set_detection_shift_horizontal(0.2);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetection(cv::Rect_<float>(.9, 0, .1, .1), 0, runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(944, 56, 111, 111, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, EmptySize) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  MP_ASSERT_OK(runner->Run());
  ASSERT_EQ(runner->Outputs().Tag(kCropRectTag).packets.size(), 0);
}

TEST(ContentZoomingCalculatorTest, EmptyDetections) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  auto input_size = ::absl::make_unique<std::pair<int, int>>(1000, 1000);
  runner->MutableInputs()
      ->Tag(kVideoSizeTag)
      .packets.push_back(Adopt(input_size.release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 1000, 1000, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ResolutionChangeStationary) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1, 500, 500,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 222, 222, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500 * 0.5, 500 * 0.5, 222 * 0.5, 222 * 0.5, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ResolutionChangeStationaryWithCache) {
  mediapipe::autoflip::ContentZoomingCalculatorStateCacheType cache;
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  config.add_input_side_packet("STATE_CACHE:state_cache");
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                          runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(500, 500, 222, 222, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1, 500, 500,
                          runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(500 * 0.5, 500 * 0.5, 222 * 0.5, 222 * 0.5, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
}

TEST(ContentZoomingCalculatorTest, ResolutionChangeZooming) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.1, .1, .8, .8), 0, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 2000000, 500, 500,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 888, 888, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 588, 588, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500 * 0.5, 500 * 0.5, 288 * 0.5, 288 * 0.5, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ResolutionChangeZoomingWithCache) {
  mediapipe::autoflip::ContentZoomingCalculatorStateCacheType cache;
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  config.add_input_side_packet("STATE_CACHE:state_cache");
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetectionFrameSize(cv::Rect_<float>(.1, .1, .8, .8), 0, 1000, 1000,
                          runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(500, 500, 888, 888, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
  // The second runner should just resume based on state from the first runner.
  {
    auto runner = ::absl::make_unique<CalculatorRunner>(config);
    runner->MutableSidePackets()->Tag(kStateCacheTag) = MakePacket<
        mediapipe::autoflip::ContentZoomingCalculatorStateCacheType*>(&cache);
    AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1000, 1000,
                          runner.get());
    AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 2000000, 500, 500,
                          runner.get());
    MP_ASSERT_OK(runner->Run());
    CheckCropRect(500, 500, 588, 588, 0,
                  runner->Outputs().Tag(kCropRectTag).packets);
    CheckCropRect(500 * 0.5, 500 * 0.5, 288 * 0.5, 288 * 0.5, 1,
                  runner->Outputs().Tag(kCropRectTag).packets);
  }
}

TEST(ContentZoomingCalculatorTest, MaxZoomValue) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_max_zoom_value_deg(55);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  // 55/60 * 1000 = 916
  CheckCropRect(500, 500, 916, 916, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, MaxZoomValueOverride) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigF);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_max_zoom_value_deg(30);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 640, 480,
                        runner.get(), {.max_zoom_factor_percent = 133});
  // Change resolution and allow more zoom, and give time to use the new limit
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1280, 720,
                        runner.get(), {.max_zoom_factor_percent = 166});
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 2000000, 1280, 720,
                        runner.get(), {.max_zoom_factor_percent = 166});
  // Switch back to a smaller resolution with a more limited zoom
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 3000000, 640, 480,
                        runner.get(), {.max_zoom_factor_percent = 133});
  MP_ASSERT_OK(runner->Run());
  // Max. 133% zoomed in means min. (100/133) ~ 75% of height left: ~360
  // Max. 166% zoomed in means min. (100/166) ~ 60% of height left: ~430
  CheckCropRect(320, 240, 480, 360, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(640, 360, 769, 433, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(320, 240, 480, 360, 3,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, MaxZoomOutValue) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_scale_factor(1.0);
  options->mutable_kinematic_options_zoom()->set_min_motion_to_reframe(5.0);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.025, .025, .95, .95), 0, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(0, 0, -1, -1), 1000000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(0, 0, -1, -1), 2000000, 1000, 1000,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  // 55/60 * 1000 = 916
  CheckCropRect(500, 500, 950, 950, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 1000, 1000, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, StartZoomedOut) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_start_zoomed_out(true);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 400000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 800000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1000, 1000,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 1000, 1000, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 880, 880, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 760, 760, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 655, 655, 3,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, AnimateToFirstRect) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_us_to_first_rect(1000000);
  options->set_us_to_first_rect_delay(500000);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 400000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 800000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1500000, 1000, 1000,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 1000, 1000, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 1000, 1000, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 470, 470, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 222, 222, 3,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 222, 222, 4,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, CanControlAnimation) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigE);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_start_zoomed_out(true);
  options->set_us_to_first_rect(1000000);
  options->set_us_to_first_rect_delay(500000);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  // Request the animation for the first frame.
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get(), {.animated_zoom = true});
  // We now stop requesting animated zoom and expect the already started
  // animation run to completion. This tests that the zoom in continues in the
  // call when it was started in the Meet greenroom.
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 400000, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 800000, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1500000, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 1000, 1000, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 1000, 1000, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 470, 470, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 222, 222, 3,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 222, 222, 4,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, DoesNotAnimateIfDisabledViaInput) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigE);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_start_zoomed_out(true);
  options->set_us_to_first_rect(1000000);
  options->set_us_to_first_rect_delay(500000);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  // Disable the animation already for the first frame.
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 400000, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 800000, 1000, 1000,
                        runner.get(), {.animated_zoom = false});
  MP_ASSERT_OK(runner->Run());
  CheckCropRect(500, 500, 1000, 1000, 0,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 880, 880, 1,
                runner->Outputs().Tag(kCropRectTag).packets);
  CheckCropRect(500, 500, 760, 760, 2,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, ProvidesZeroSizeFirstRectWithoutDetections) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);

  auto input_size = ::absl::make_unique<std::pair<int, int>>(1000, 1000);
  runner->MutableInputs()
      ->Tag(kVideoSizeTag)
      .packets.push_back(Adopt(input_size.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kFirstCropRectTag).packets;
  ASSERT_EQ(output_packets.size(), 1);
  const auto& rect = output_packets[0].Get<mediapipe::NormalizedRect>();
  EXPECT_EQ(rect.x_center(), 0);
  EXPECT_EQ(rect.y_center(), 0);
  EXPECT_EQ(rect.width(), 0);
  EXPECT_EQ(rect.height(), 0);
}

TEST(ContentZoomingCalculatorTest, ProvidesConstantFirstRect) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_us_to_first_rect(500000);
  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 0, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 500000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1000000, 1000, 1000,
                        runner.get());
  AddDetectionFrameSize(cv::Rect_<float>(.4, .4, .2, .2), 1500000, 1000, 1000,
                        runner.get());
  MP_ASSERT_OK(runner->Run());
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kFirstCropRectTag).packets;
  ASSERT_EQ(output_packets.size(), 4);
  const auto& first_rect = output_packets[0].Get<mediapipe::NormalizedRect>();
  EXPECT_NEAR(first_rect.x_center(), 0.5, 0.05);
  EXPECT_NEAR(first_rect.y_center(), 0.5, 0.05);
  EXPECT_NEAR(first_rect.width(), 0.222, 0.05);
  EXPECT_NEAR(first_rect.height(), 0.222, 0.05);
  for (int i = 1; i < 4; ++i) {
    const auto& rect = output_packets[i].Get<mediapipe::NormalizedRect>();
    EXPECT_EQ(first_rect.x_center(), rect.x_center());
    EXPECT_EQ(first_rect.y_center(), rect.y_center());
    EXPECT_EQ(first_rect.width(), rect.width());
    EXPECT_EQ(first_rect.height(), rect.height());
  }
}

TEST(ContentZoomingCalculatorTest, AllowsCroppingOutsideFrame) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_allow_cropping_outside_frame(true);
  auto runner = ::std::make_unique<CalculatorRunner>(config);

  AddDetection(cv::Rect_<float>(-0.5, -0.5, 1.0, 1.0), 0, runner.get());
  MP_ASSERT_OK(runner->Run());

  CheckCropRect(/* x_center= */ 0, /* y_center= */ 0, /* width= */ 1000,
                /* height= */ 1000, /* frame_number= */ 0,
                runner->Outputs().Tag(kCropRectTag).packets);
}

TEST(ContentZoomingCalculatorTest, InitialEmptyDetectionDefaultsToNoCrop) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigD);
  auto* options = config.mutable_options()->MutableExtension(
      ContentZoomingCalculatorOptions::ext);
  options->set_allow_cropping_outside_frame(true);
  auto runner = ::std::make_unique<CalculatorRunner>(config);
  int64_t time = 0;
  int width = 1000;
  int height = 1000;

  auto empty_detections = std::make_unique<std::vector<mediapipe::Detection>>();
  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(empty_detections.release()).At(Timestamp(time)));
  auto input_size = ::std::make_unique<std::pair<int, int>>(width, height);
  runner->MutableInputs()
      ->Tag("VIDEO_SIZE")
      .packets.push_back(Adopt(input_size.release()).At(Timestamp(time)));
  MP_ASSERT_OK(runner->Run());

  CheckCropRect(/* x_center= */ 500, /* y_center= */ 500, width, height,
                /* frame_number= */ 0,
                runner->Outputs().Tag(kCropRectTag).packets);
}

}  // namespace
}  // namespace autoflip

}  // namespace mediapipe
