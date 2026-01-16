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

#include "mediapipe/examples/desktop/autoflip/calculators/scene_cropping_calculator.h"

#include <random>
#include <utility>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
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

constexpr char kFramingDetectionsVizFramesTag[] =
    "FRAMING_DETECTIONS_VIZ_FRAMES";
constexpr char kExternalRenderingFullVidTag[] = "EXTERNAL_RENDERING_FULL_VID";
constexpr char kExternalRenderingPerFrameTag[] = "EXTERNAL_RENDERING_PER_FRAME";
constexpr char kCroppingSummaryTag[] = "CROPPING_SUMMARY";
constexpr char kSalientPointFrameVizFramesTag[] =
    "SALIENT_POINT_FRAME_VIZ_FRAMES";
constexpr char kKeyFrameCropRegionVizFramesTag[] =
    "KEY_FRAME_CROP_REGION_VIZ_FRAMES";
constexpr char kCroppedFramesTag[] = "CROPPED_FRAMES";
constexpr char kShotBoundariesTag[] = "SHOT_BOUNDARIES";
constexpr char kStaticFeaturesTag[] = "STATIC_FEATURES";
constexpr char kVideoSizeTag[] = "VIDEO_SIZE";
constexpr char kVideoFramesTag[] = "VIDEO_FRAMES";
constexpr char kDetectionFeaturesTag[] = "DETECTION_FEATURES";
constexpr char kKeyFramesTag[] = "KEY_FRAMES";

using ::testing::HasSubstr;

constexpr char kConfig[] = R"(
  calculator: "SceneCroppingCalculator"
  input_stream: "VIDEO_FRAMES:camera_frames_org"
  input_stream: "KEY_FRAMES:down_sampled_frames"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:border_features"
  input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
  output_stream: "CROPPED_FRAMES:cropped_frames"
  options: {
    [mediapipe.autoflip.SceneCroppingCalculatorOptions.ext]: {
      target_width: $0
      target_height: $1
      target_size_type: $2
      max_scene_size: $3
      prior_frame_buffer_size: $4
    }
  })";

constexpr char kNoKeyFrameConfig[] = R"(
  calculator: "SceneCroppingCalculator"
  input_stream: "VIDEO_FRAMES:camera_frames_org"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:border_features"
  input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
  output_stream: "CROPPED_FRAMES:cropped_frames"
  options: {
    [mediapipe.autoflip.SceneCroppingCalculatorOptions.ext]: {
      target_width: $0
      target_height: $1
    }
  })";

constexpr char kDebugConfigNoCroppedFrame[] = R"(
  calculator: "SceneCroppingCalculator"
  input_stream: "VIDEO_FRAMES:camera_frames_org"
  input_stream: "KEY_FRAMES:down_sampled_frames"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:border_features"
  input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
  output_stream: "KEY_FRAME_CROP_REGION_VIZ_FRAMES:key_frame_crop_viz_frames"
  output_stream: "SALIENT_POINT_FRAME_VIZ_FRAMES:salient_point_viz_frames"
  options: {
    [mediapipe.autoflip.SceneCroppingCalculatorOptions.ext]: {
      target_width: $0
      target_height: $1
    }
  })";

constexpr char kDebugConfig[] = R"(
  calculator: "SceneCroppingCalculator"
  input_stream: "VIDEO_FRAMES:camera_frames_org"
  input_stream: "KEY_FRAMES:down_sampled_frames"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:border_features"
  input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
  output_stream: "CROPPED_FRAMES:cropped_frames"
  output_stream: "KEY_FRAME_CROP_REGION_VIZ_FRAMES:key_frame_crop_viz_frames"
  output_stream: "SALIENT_POINT_FRAME_VIZ_FRAMES:salient_point_viz_frames"
  output_stream: "FRAMING_DETECTIONS_VIZ_FRAMES:framing_viz_frames"
  output_stream: "CROPPING_SUMMARY:cropping_summaries"
  output_stream: "EXTERNAL_RENDERING_PER_FRAME:external_rendering_per_frame"
  output_stream: "EXTERNAL_RENDERING_FULL_VID:external_rendering_full_vid"
  options: {
    [mediapipe.autoflip.SceneCroppingCalculatorOptions.ext]: {
      target_width: $0
      target_height: $1
    }
  })";

constexpr char kExternalRenderConfig[] = R"(
  calculator: "SceneCroppingCalculator"
  input_stream: "VIDEO_FRAMES:camera_frames_org"
  input_stream: "KEY_FRAMES:down_sampled_frames"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:border_features"
  input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
  output_stream: "EXTERNAL_RENDERING_PER_FRAME:external_rendering_per_frame"
  output_stream: "EXTERNAL_RENDERING_FULL_VID:external_rendering_full_vid"
  options: {
    [mediapipe.autoflip.SceneCroppingCalculatorOptions.ext]: {
      target_width: $0
      target_height: $1
    }
  })";

constexpr char kExternalRenderConfigNoVideo[] = R"(
  calculator: "SceneCroppingCalculator"
  input_stream: "VIDEO_SIZE:camera_size"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:border_features"
  input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
  output_stream: "EXTERNAL_RENDERING_PER_FRAME:external_rendering_per_frame"
  output_stream: "EXTERNAL_RENDERING_FULL_VID:external_rendering_full_vid"
  options: {
    [mediapipe.autoflip.SceneCroppingCalculatorOptions.ext]: {
      target_width: $0
      target_height: $1
      video_features_width: $2
      video_features_height: $3
    }
  })";

constexpr int kInputFrameWidth = 1280;
constexpr int kInputFrameHeight = 720;

constexpr int kKeyFrameWidth = 640;
constexpr int kKeyFrameHeight = 360;

constexpr int kTargetWidth = 720;
constexpr int kTargetHeight = 1124;
constexpr SceneCroppingCalculatorOptions::TargetSizeType kTargetSizeType =
    SceneCroppingCalculatorOptions::USE_TARGET_DIMENSION;

constexpr int kNumScenes = 3;
constexpr int kSceneSize = 8;
constexpr int kMaxSceneSize = 10;
constexpr int kPriorFrameBufferSize = 5;

constexpr int kMinNumDetections = 0;
constexpr int kMaxNumDetections = 10;

constexpr int kDownSampleRate = 4;
constexpr int64_t kTimestampDiff = 20000;

// Returns a singleton random engine for generating random values. The seed is
// fixed for reproducibility.
std::default_random_engine& GetGen() {
  static std::default_random_engine generator{0};
  return generator;
}

// Returns random color with r, g, b in the range of [0, 255].
cv::Scalar GetRandomColor() {
  std::uniform_int_distribution<int> distribution(0, 255);
  const int red = distribution(GetGen());
  const int green = distribution(GetGen());
  const int blue = distribution(GetGen());
  return cv::Scalar(red, green, blue);
}

// Makes a detection set given number of detections. Each detection has randomly
// generated regions within given width and height with random score in [0, 1],
// and is randomly set to be required or non-required.
std::unique_ptr<DetectionSet> MakeDetections(const int num_detections,
                                             const int width,
                                             const int height) {
  std::uniform_int_distribution<int> width_distribution(0, width);
  std::uniform_int_distribution<int> height_distribution(0, height);
  std::uniform_real_distribution<float> score_distribution(0.0, 1.0);
  std::bernoulli_distribution is_required_distribution(0.5);
  auto detections = absl::make_unique<DetectionSet>();
  for (int i = 0; i < num_detections; ++i) {
    auto* region = detections->add_detections();
    const int x1 = width_distribution(GetGen());
    const int x2 = width_distribution(GetGen());
    const int y1 = height_distribution(GetGen());
    const int y2 = height_distribution(GetGen());
    const int x_min = std::min(x1, x2), x_max = std::max(x1, x2);
    const int y_min = std::min(y1, y2), y_max = std::max(y1, y2);
    auto* location = region->mutable_location();
    location->set_x(x_min);
    location->set_width(x_max - x_min);
    location->set_y(y_min);
    location->set_height(y_max - y_min);
    region->set_score(score_distribution(GetGen()));
    region->set_is_required(is_required_distribution(GetGen()));
  }
  return detections;
}

// Makes a detection set given number of detections. Each detection has randomly
// generated regions within given width and height with random score in [0, 1],
// and is randomly set to be required or non-required.
std::unique_ptr<DetectionSet> MakeCenterDetection(const int width,
                                                  const int height) {
  auto detections = absl::make_unique<DetectionSet>();
  auto* region = detections->add_detections();
  auto* location = region->mutable_location();
  location->set_x(width / 2 - 5);
  location->set_width(width / 2 + 10);
  location->set_y(height / 2 - 5);
  location->set_height(height);
  region->set_score(1);
  return detections;
}

// Makes an image frame of solid color given color, width, and height.
std::unique_ptr<ImageFrame> MakeImageFrameFromColor(const cv::Scalar& color,
                                                    const int width,
                                                    const int height) {
  auto image_frame =
      absl::make_unique<ImageFrame>(ImageFormat::SRGB, width, height);
  auto mat = formats::MatView(image_frame.get());
  mat = color;
  return image_frame;
}

// Adds key frame detection features given time (in ms) to the input stream.
// Randomly generates a number of detections in the range of kMinNumDetections
// and kMaxNumDetections. Optionally add a key image frame of random solid color
// and given size.
void AddKeyFrameFeatures(const int64_t time_ms, const int key_frame_width,
                         const int key_frame_height, bool randomize,
                         CalculatorRunner::StreamContentsSet* inputs) {
  Timestamp timestamp(time_ms);
  if (inputs->HasTag(kKeyFramesTag)) {
    auto key_frame = MakeImageFrameFromColor(GetRandomColor(), key_frame_width,
                                             key_frame_height);
    inputs->Tag(kKeyFramesTag)
        .packets.push_back(Adopt(key_frame.release()).At(timestamp));
  }
  if (randomize) {
    const int num_detections = std::uniform_int_distribution<int>(
        kMinNumDetections, kMaxNumDetections)(GetGen());
    auto detections =
        MakeDetections(num_detections, key_frame_width, key_frame_height);
    inputs->Tag(kDetectionFeaturesTag)
        .packets.push_back(Adopt(detections.release()).At(timestamp));
  } else {
    auto detections = MakeCenterDetection(key_frame_width, key_frame_height);
    inputs->Tag(kDetectionFeaturesTag)
        .packets.push_back(Adopt(detections.release()).At(timestamp));
  }
}

// Adds a scene given number of frames to the input stream. Spaces frame at the
// default timestamp interval starting from given start frame index. Scene has
// empty static features.
void AddScene(const int start_frame_index, const int num_scene_frames,
              const int frame_width, const int frame_height,
              const int key_frame_width, const int key_frame_height,
              const int DownSampleRate,
              CalculatorRunner::StreamContentsSet* inputs) {
  int64_t time_ms = start_frame_index * kTimestampDiff;
  for (int i = 0; i < num_scene_frames; ++i) {
    Timestamp timestamp(time_ms);
    if (inputs->HasTag(kVideoFramesTag)) {
      auto frame =
          MakeImageFrameFromColor(GetRandomColor(), frame_width, frame_height);
      inputs->Tag(kVideoFramesTag)
          .packets.push_back(Adopt(frame.release()).At(timestamp));
    } else {
      auto input_size =
          ::absl::make_unique<std::pair<int, int>>(frame_width, frame_height);
      inputs->Tag(kVideoSizeTag)
          .packets.push_back(Adopt(input_size.release()).At(timestamp));
    }
    auto static_features = absl::make_unique<StaticFeatures>();
    inputs->Tag(kStaticFeaturesTag)
        .packets.push_back(Adopt(static_features.release()).At(timestamp));
    if (DownSampleRate == 1) {
      AddKeyFrameFeatures(time_ms, key_frame_width, key_frame_height, false,
                          inputs);
    } else if (i % DownSampleRate == 0) {  // is a key frame
      AddKeyFrameFeatures(time_ms, key_frame_width, key_frame_height, true,
                          inputs);
    }
    if (i == num_scene_frames - 1) {  // adds shot boundary
      inputs->Tag(kShotBoundariesTag)
          .packets.push_back(Adopt(new bool(true)).At(Timestamp(time_ms)));
    }
    time_ms += kTimestampDiff;
  }
}

// Checks that the output stream for cropped frames has the correct number of
// frames, and that the size of each frame is correct.
void CheckCroppedFrames(const CalculatorRunner& runner, const int num_frames,
                        const int target_width, const int target_height) {
  const auto& outputs = runner.Outputs();
  EXPECT_TRUE(outputs.HasTag(kCroppedFramesTag));
  const auto& cropped_frames_outputs = outputs.Tag(kCroppedFramesTag).packets;
  EXPECT_EQ(cropped_frames_outputs.size(), num_frames);
  for (int i = 0; i < num_frames; ++i) {
    const auto& cropped_frame = cropped_frames_outputs[i].Get<ImageFrame>();
    EXPECT_EQ(cropped_frame.Width(), target_width);
    EXPECT_EQ(cropped_frame.Height(), target_height);
  }
}

// Checks that the calculator checks the maximum scene size is valid.
TEST(SceneCroppingCalculatorTest, ChecksMaxSceneSize) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kConfig, kTargetWidth, kTargetHeight,
                           kTargetSizeType, 0, kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const auto status = runner->Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Maximum scene size is non-positive."));
}

// Checks that the calculator checks the prior frame buffer size is valid.
TEST(SceneCroppingCalculatorTest, ChecksPriorFrameBufferSize) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kConfig, kTargetWidth, kTargetHeight,
                           kTargetSizeType, kMaxSceneSize, -1));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const auto status = runner->Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Prior frame buffer size is negative."));
}

TEST(SceneCroppingCalculatorTest, ChecksDebugConfigWithoutCroppedFrame) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kDebugConfigNoCroppedFrame, kTargetWidth, kTargetHeight));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const auto status = runner->Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("can only be used when"));
}

// Checks that the calculator crops scene frames when there is no input key
// frames stream.
TEST(SceneCroppingCalculatorTest, HandlesNoKeyFrames) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kNoKeyFrameConfig, kTargetWidth, kTargetHeight));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  AddScene(0, kSceneSize, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, kDownSampleRate, runner->MutableInputs());
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, kSceneSize, kTargetWidth, kTargetHeight);
}

// Checks that the calculator handles scenes longer than maximum scene size (
// force flush is triggered).
TEST(SceneCroppingCalculatorTest, HandlesLongScene) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, kTargetWidth, kTargetHeight, kTargetSizeType, kMaxSceneSize,
          kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  AddScene(0, 2 * kMaxSceneSize, kInputFrameWidth, kInputFrameHeight,
           kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
           runner->MutableInputs());
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, 2 * kMaxSceneSize, kTargetWidth, kTargetHeight);
}

// Checks that the calculator can optionally output debug streams.
TEST(SceneCroppingCalculatorTest, OutputsDebugStreams) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kDebugConfig, kTargetWidth, kTargetHeight));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const int num_frames = kSceneSize;
  AddScene(0, num_frames, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, kDownSampleRate, runner->MutableInputs());

  MP_EXPECT_OK(runner->Run());
  const auto& outputs = runner->Outputs();
  EXPECT_TRUE(outputs.HasTag(kKeyFrameCropRegionVizFramesTag));
  EXPECT_TRUE(outputs.HasTag(kSalientPointFrameVizFramesTag));
  EXPECT_TRUE(outputs.HasTag(kCroppingSummaryTag));
  EXPECT_TRUE(outputs.HasTag(kExternalRenderingPerFrameTag));
  EXPECT_TRUE(outputs.HasTag(kExternalRenderingFullVidTag));
  EXPECT_TRUE(outputs.HasTag(kFramingDetectionsVizFramesTag));
  const auto& crop_region_viz_frames_outputs =
      outputs.Tag(kKeyFrameCropRegionVizFramesTag).packets;
  const auto& salient_point_viz_frames_outputs =
      outputs.Tag(kSalientPointFrameVizFramesTag).packets;
  const auto& summary_output = outputs.Tag(kCroppingSummaryTag).packets;
  const auto& ext_render_per_frame =
      outputs.Tag(kExternalRenderingPerFrameTag).packets;
  const auto& ext_render_full_vid =
      outputs.Tag(kExternalRenderingFullVidTag).packets;
  const auto& framing_viz_frames_output =
      outputs.Tag(kFramingDetectionsVizFramesTag).packets;
  EXPECT_EQ(crop_region_viz_frames_outputs.size(), num_frames);
  EXPECT_EQ(salient_point_viz_frames_outputs.size(), num_frames);
  EXPECT_EQ(framing_viz_frames_output.size(), num_frames);
  EXPECT_EQ(summary_output.size(), 1);
  EXPECT_EQ(ext_render_per_frame.size(), num_frames);
  EXPECT_EQ(ext_render_full_vid.size(), 1);
  EXPECT_EQ(ext_render_per_frame[0].Get<ExternalRenderFrame>().timestamp_us(),
            0);
  EXPECT_EQ(ext_render_full_vid[0]
                .Get<std::vector<ExternalRenderFrame>>()[0]
                .timestamp_us(),
            0);
  EXPECT_EQ(ext_render_per_frame[1].Get<ExternalRenderFrame>().timestamp_us(),
            20000);
  EXPECT_EQ(ext_render_full_vid[0]
                .Get<std::vector<ExternalRenderFrame>>()[1]
                .timestamp_us(),
            20000);

  for (int i = 0; i < num_frames; ++i) {
    const auto& crop_region_viz_frame =
        crop_region_viz_frames_outputs[i].Get<ImageFrame>();
    EXPECT_EQ(crop_region_viz_frame.Width(), kInputFrameWidth);
    EXPECT_EQ(crop_region_viz_frame.Height(), kInputFrameHeight);
    const auto& salient_point_viz_frame =
        salient_point_viz_frames_outputs[i].Get<ImageFrame>();
    EXPECT_EQ(salient_point_viz_frame.Width(), kInputFrameWidth);
    EXPECT_EQ(salient_point_viz_frame.Height(), kInputFrameHeight);
  }
  const auto& summary = summary_output[0].Get<VideoCroppingSummary>();
  EXPECT_EQ(summary.scene_summaries_size(), 2);
  const auto& summary_0 = summary.scene_summaries(0);
  EXPECT_TRUE(summary_0.is_padded());
  EXPECT_TRUE(summary_0.camera_motion().has_steady_motion());
}

// Checks that the calculator handles the case of generating landscape frames.
TEST(SceneCroppingCalculatorTest, HandlesLandscapeTarget) {
  const int input_width = 900;
  const int input_height = 1600;
  const int target_width = 1200;
  const int target_height = 800;
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, target_width, target_height, kTargetSizeType, kMaxSceneSize,
          kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  for (int i = 0; i < kNumScenes; ++i) {
    AddScene(i * kSceneSize, kSceneSize, input_width, input_height,
             kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
             runner->MutableInputs());
  }
  const int num_frames = kSceneSize * kNumScenes;
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, num_frames, target_width, target_height);
}

// Checks that the calculator crops scene frames to target size when the target
// size type is the default USE_TARGET_DIMENSION.
TEST(SceneCroppingCalculatorTest, CropsToTargetSize) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, kTargetWidth, kTargetHeight, kTargetSizeType, kMaxSceneSize,
          kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  for (int i = 0; i < kNumScenes; ++i) {
    AddScene(i * kSceneSize, kSceneSize, kInputFrameWidth, kInputFrameHeight,
             kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
             runner->MutableInputs());
  }
  const int num_frames = kSceneSize * kNumScenes;
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, num_frames, kTargetWidth, kTargetHeight);
}

// Checks that the calculator crops scene frames to input size when the target
// size type is KEEP_ORIGINAL_DIMENSION.
TEST(SceneCroppingCalculatorTest, CropsToOriginalDimension) {
  // target_width and target_height are ignored
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, /*target_width*/ 2, /*target_height*/ 2,
          SceneCroppingCalculatorOptions::KEEP_ORIGINAL_DIMENSION,
          kMaxSceneSize, kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  for (int i = 0; i < kNumScenes; ++i) {
    AddScene(i * kSceneSize, kSceneSize, kInputFrameWidth, kInputFrameHeight,
             kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
             runner->MutableInputs());
  }
  const int num_frames = kSceneSize * kNumScenes;
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, num_frames, kInputFrameWidth, kInputFrameHeight);
}

// Checks that the calculator keeps original height if the target size type is
// set to KEEP_ORIGINAL_HEIGHT.
TEST(SceneCroppingCalculatorTest, KeepsOriginalHeight) {
  const auto target_size_type =
      SceneCroppingCalculatorOptions::KEEP_ORIGINAL_HEIGHT;
  const int target_height = kInputFrameHeight;
  const double target_aspect_ratio =
      static_cast<double>(kTargetWidth) / kTargetHeight;
  int target_width = std::round(target_height * target_aspect_ratio);
  if (target_width % 2 == 1) target_width--;
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, kTargetWidth, kTargetHeight, target_size_type, kMaxSceneSize,
          kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  AddScene(0, kMaxSceneSize, kInputFrameWidth, kInputFrameHeight,
           kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
           runner->MutableInputs());
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, kMaxSceneSize, target_width, target_height);
}

// Checks that the calculator keeps original width if the target size type is
// set to KEEP_ORIGINAL_WIDTH.
TEST(SceneCroppingCalculatorTest, KeepsOriginalWidth) {
  const auto target_size_type =
      SceneCroppingCalculatorOptions::KEEP_ORIGINAL_WIDTH;
  const int target_width = kInputFrameWidth;
  const double target_aspect_ratio =
      static_cast<double>(kTargetWidth) / kTargetHeight;
  int target_height = std::round(target_width / target_aspect_ratio);
  if (target_height % 2 == 1) target_height--;
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, kTargetWidth, kTargetHeight, target_size_type, kMaxSceneSize,
          kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  AddScene(0, kMaxSceneSize, kInputFrameWidth, kInputFrameHeight,
           kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
           runner->MutableInputs());
  MP_EXPECT_OK(runner->Run());
  CheckCroppedFrames(*runner, kMaxSceneSize, target_width, target_height);
}

// Checks that the calculator rejects odd target size.
TEST(SceneCroppingCalculatorTest, RejectsOddTargetSize) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          kConfig, kTargetWidth - 1, kTargetHeight, kTargetSizeType,
          kMaxSceneSize, kPriorFrameBufferSize));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  AddScene(0, kMaxSceneSize, kInputFrameWidth, kInputFrameHeight,
           kKeyFrameWidth, kKeyFrameHeight, kDownSampleRate,
           runner->MutableInputs());
  const auto status = runner->Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Target width cannot be odd"));
}

// Checks that the calculator always produces even frame size given even input
// frame size and even target under all target size types.
TEST(SceneCroppingCalculatorTest, ProducesEvenFrameSize) {
  // Some commonly used video resolution (some are divided by 10 to make the
  // test faster), and some odd input frame sizes.
  const std::vector<std::pair<int, int>> video_sizes = {
      {384, 216}, {256, 144}, {192, 108}, {128, 72},  {640, 360},
      {426, 240}, {100, 100}, {214, 100}, {240, 100}, {720, 1124},
      {90, 160},  {641, 360}, {640, 361}, {101, 101}};

  const std::vector<SceneCroppingCalculatorOptions::TargetSizeType>
      target_size_types = {SceneCroppingCalculatorOptions::USE_TARGET_DIMENSION,
                           SceneCroppingCalculatorOptions::KEEP_ORIGINAL_HEIGHT,
                           SceneCroppingCalculatorOptions::KEEP_ORIGINAL_WIDTH};

  // Exhaustive check on each size as input and each size as output for each
  // target size type.
  for (int i = 0; i < video_sizes.size(); ++i) {
    const int frame_width = video_sizes[i].first;
    const int frame_height = video_sizes[i].second;
    for (int j = 0; j < video_sizes.size(); ++j) {
      const int target_width = video_sizes[j].first;
      const int target_height = video_sizes[j].second;
      if (target_width % 2 == 1 || target_height % 2 == 1) continue;
      for (int k = 0; k < target_size_types.size(); ++k) {
        const CalculatorGraphConfig::Node config =
            ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
                kConfig, target_width, target_height, target_size_types[k],
                kMaxSceneSize, kPriorFrameBufferSize));
        auto runner = absl::make_unique<CalculatorRunner>(config);
        AddScene(0, 1, frame_width, frame_height, kKeyFrameWidth,
                 kKeyFrameHeight, kDownSampleRate, runner->MutableInputs());
        MP_EXPECT_OK(runner->Run());
        const auto& output_frame = runner->Outputs()
                                       .Tag(kCroppedFramesTag)
                                       .packets[0]
                                       .Get<ImageFrame>();
        EXPECT_EQ(output_frame.Width() % 2, 0);
        EXPECT_EQ(output_frame.Height() % 2, 0);
        if (target_size_types[k] ==
            SceneCroppingCalculatorOptions::USE_TARGET_DIMENSION) {
          EXPECT_EQ(output_frame.Width(), target_width);
          EXPECT_EQ(output_frame.Height(), target_height);
        } else if (target_size_types[k] ==
                   SceneCroppingCalculatorOptions::KEEP_ORIGINAL_HEIGHT) {
          // Difference could be 1 if input size is odd.
          EXPECT_LE(std::abs(output_frame.Height() - frame_height), 1);
        } else if (target_size_types[k] ==
                   SceneCroppingCalculatorOptions::KEEP_ORIGINAL_WIDTH) {
          EXPECT_LE(std::abs(output_frame.Width() - frame_width), 1);
        }
      }
    }
  }
}

// Checks that the calculator pads the frames with solid color when possible.
TEST(SceneCroppingCalculatorTest, PadsWithSolidColorFromStaticFeatures) {
  const int target_width = 100, target_height = 200;
  const int input_width = 100, input_height = 100;
  CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kNoKeyFrameConfig, target_width, target_height));
  auto* options = config.mutable_options()->MutableExtension(
      SceneCroppingCalculatorOptions::ext);
  options->set_solid_background_frames_padding_fraction(0.6);
  auto runner = absl::make_unique<CalculatorRunner>(config);

  const int static_features_downsample_rate = 2;
  const float fraction_with_solid_background = 0.7;
  const int red = 122, green = 167, blue = 250;
  const int num_frames_with_solid_background =
      std::round(fraction_with_solid_background * kSceneSize /
                 static_features_downsample_rate);

  // Add inputs.
  auto* inputs = runner->MutableInputs();
  int64_t time_ms = 0;
  int num_static_features = 0;
  for (int i = 0; i < kSceneSize; ++i) {
    Timestamp timestamp(time_ms);
    auto frame =
        MakeImageFrameFromColor(GetRandomColor(), input_width, input_height);
    inputs->Tag(kVideoFramesTag)
        .packets.push_back(Adopt(frame.release()).At(timestamp));
    if (i % static_features_downsample_rate == 0) {
      auto static_features = absl::make_unique<StaticFeatures>();
      if (num_static_features < num_frames_with_solid_background) {
        auto* color = static_features->mutable_solid_background();
        // Uses BGR to mimic input from static features solid background color.
        color->set_r(blue);
        color->set_g(green);
        color->set_b(red);
      }
      inputs->Tag(kStaticFeaturesTag)
          .packets.push_back(Adopt(static_features.release()).At(timestamp));
      num_static_features++;
    }
    if (i % kDownSampleRate == 0) {  // is a key frame
      // Target crop size is (50, 100). Adds one required detection with size
      // (80, 100) larger than the target crop size to force padding.
      auto detections = absl::make_unique<DetectionSet>();
      auto* salient_region = detections->add_detections();
      salient_region->set_is_required(true);
      auto* location = salient_region->mutable_location();
      location->set_x(10);
      location->set_y(0);
      location->set_width(80);
      location->set_height(input_height);
      inputs->Tag(kDetectionFeaturesTag)
          .packets.push_back(Adopt(detections.release()).At(timestamp));
    }
    time_ms += kTimestampDiff;
  }

  MP_EXPECT_OK(runner->Run());

  // Checks that the top and bottom borders indeed have the background color.
  const int border_size = 37;
  const auto& cropped_frames_outputs =
      runner->Outputs().Tag(kCroppedFramesTag).packets;
  EXPECT_EQ(cropped_frames_outputs.size(), kSceneSize);
  for (int i = 0; i < kSceneSize; ++i) {
    const auto& cropped_frame = cropped_frames_outputs[i].Get<ImageFrame>();
    cv::Mat mat = formats::MatView(&cropped_frame);
    for (int x = 0; x < target_width; ++x) {
      for (int y = 0; y < border_size; ++y) {
        EXPECT_EQ(mat.at<cv::Vec3b>(y, x)[0], red);
        EXPECT_EQ(mat.at<cv::Vec3b>(y, x)[1], green);
        EXPECT_EQ(mat.at<cv::Vec3b>(y, x)[2], blue);
      }
      for (int y2 = 0; y2 < border_size; ++y2) {
        const int y = target_height - 1 - y2;
        EXPECT_EQ(mat.at<cv::Vec3b>(y, x)[0], red);
        EXPECT_EQ(mat.at<cv::Vec3b>(y, x)[1], green);
        EXPECT_EQ(mat.at<cv::Vec3b>(y, x)[2], blue);
      }
    }
  }
}

// Checks that the calculator removes static borders from frames.
TEST(SceneCroppingCalculatorTest, RemovesStaticBorders) {
  const int target_width = 50, target_height = 100;
  const int input_width = 100, input_height = 100;
  const int top_border_size = 20, bottom_border_size = 20;
  const cv::Rect top_border_rect(0, 0, input_width, top_border_size);
  const cv::Rect bottom_border_rect(0, input_height - bottom_border_size,
                                    input_width, bottom_border_size);
  const cv::Scalar frame_color = cv::Scalar(255, 255, 255);
  const cv::Scalar border_color = cv::Scalar(0, 0, 0);

  const auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::Substitute(kNoKeyFrameConfig, target_width, target_height));
  auto runner = absl::make_unique<CalculatorRunner>(config);

  // Add inputs.
  auto* inputs = runner->MutableInputs();
  const auto timestamp = Timestamp(0);
  // Make frame with borders.
  auto frame = MakeImageFrameFromColor(frame_color, input_width, input_height);
  auto mat = formats::MatView(frame.get());
  mat(top_border_rect) = border_color;
  mat(bottom_border_rect) = border_color;
  inputs->Tag(kVideoFramesTag)
      .packets.push_back(Adopt(frame.release()).At(timestamp));
  // Set borders in static features.
  auto static_features = absl::make_unique<StaticFeatures>();
  auto* top_part = static_features->add_border();
  top_part->set_relative_position(Border::TOP);
  top_part->mutable_border_position()->set_height(top_border_size);
  auto* bottom_part = static_features->add_border();
  bottom_part->set_relative_position(Border::BOTTOM);
  bottom_part->mutable_border_position()->set_height(bottom_border_size);
  inputs->Tag(kStaticFeaturesTag)
      .packets.push_back(Adopt(static_features.release()).At(timestamp));
  // Add empty detections to ensure no padding is used.
  auto detections = absl::make_unique<DetectionSet>();
  inputs->Tag(kDetectionFeaturesTag)
      .packets.push_back(Adopt(detections.release()).At(timestamp));

  MP_EXPECT_OK(runner->Run());

  // Checks that the top and bottom borders are removed. Each frame should have
  // solid color equal to frame color.
  const auto& cropped_frames_outputs =
      runner->Outputs().Tag(kCroppedFramesTag).packets;
  EXPECT_EQ(cropped_frames_outputs.size(), 1);
  const auto& cropped_frame = cropped_frames_outputs[0].Get<ImageFrame>();
  const auto cropped_mat = formats::MatView(&cropped_frame);
  for (int x = 0; x < target_width; ++x) {
    for (int y = 0; y < target_height; ++y) {
      EXPECT_EQ(cropped_mat.at<cv::Vec3b>(y, x)[0], frame_color[0]);
      EXPECT_EQ(cropped_mat.at<cv::Vec3b>(y, x)[1], frame_color[1]);
      EXPECT_EQ(cropped_mat.at<cv::Vec3b>(y, x)[2], frame_color[2]);
    }
  }
}

// Checks external render message with default poly path solver.
TEST(SceneCroppingCalculatorTest, OutputsCropMessagePolyPath) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kExternalRenderConfig, kTargetWidth, kTargetHeight));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const int num_frames = kSceneSize;
  AddScene(0, num_frames, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, 1, runner->MutableInputs());

  MP_EXPECT_OK(runner->Run());
  const auto& outputs = runner->Outputs();
  const auto& ext_render_per_frame =
      outputs.Tag(kExternalRenderingPerFrameTag).packets;
  EXPECT_EQ(ext_render_per_frame.size(), num_frames);

  for (int i = 0; i < num_frames - 1; ++i) {
    const auto& ext_render_message =
        ext_render_per_frame[i].Get<ExternalRenderFrame>();
    EXPECT_EQ(ext_render_message.timestamp_us(), i * 20000);
    EXPECT_EQ(ext_render_message.crop_from_location().x(), 725);
    EXPECT_EQ(ext_render_message.crop_from_location().y(), 0);
    EXPECT_EQ(ext_render_message.crop_from_location().width(), 461);
    EXPECT_EQ(ext_render_message.crop_from_location().height(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().x(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().y(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().width(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().height(), 1124);
  }
}

// Checks external render message with kinematic path solver.
TEST(SceneCroppingCalculatorTest, OutputsCropMessageKinematicPath) {
  CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kDebugConfig, kTargetWidth, kTargetHeight));
  auto* options = config.mutable_options()->MutableExtension(
      SceneCroppingCalculatorOptions::ext);
  auto* kinematic_options =
      options->mutable_camera_motion_options()->mutable_kinematic_options();
  kinematic_options->set_min_motion_to_reframe(1.2);
  kinematic_options->set_max_velocity(200);

  auto runner = absl::make_unique<CalculatorRunner>(config);
  const int num_frames = kSceneSize;
  AddScene(0, num_frames, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, 1, runner->MutableInputs());

  MP_EXPECT_OK(runner->Run());
  const auto& outputs = runner->Outputs();
  const auto& ext_render_per_frame =
      outputs.Tag(kExternalRenderingPerFrameTag).packets;
  EXPECT_EQ(ext_render_per_frame.size(), num_frames);

  for (int i = 0; i < num_frames - 1; ++i) {
    const auto& ext_render_message =
        ext_render_per_frame[i].Get<ExternalRenderFrame>();
    EXPECT_EQ(ext_render_message.timestamp_us(), i * 20000);
    EXPECT_EQ(ext_render_message.crop_from_location().x(), 725);
    EXPECT_EQ(ext_render_message.crop_from_location().y(), 0);
    EXPECT_EQ(ext_render_message.crop_from_location().width(), 461);
    EXPECT_EQ(ext_render_message.crop_from_location().height(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().x(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().y(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().width(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().height(), 1124);
  }
}

// Checks external render message with default poly path solver without video
// input.
TEST(SceneCroppingCalculatorTest, OutputsCropMessagePolyPathNoVideo) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kExternalRenderConfigNoVideo, kTargetWidth,
                           kTargetHeight, kKeyFrameWidth, kKeyFrameHeight));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const int num_frames = kSceneSize;
  AddScene(0, num_frames, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, 1, runner->MutableInputs());

  MP_EXPECT_OK(runner->Run());
  const auto& outputs = runner->Outputs();
  const auto& ext_render_per_frame =
      outputs.Tag(kExternalRenderingPerFrameTag).packets;
  EXPECT_EQ(ext_render_per_frame.size(), num_frames);

  for (int i = 0; i < num_frames - 1; ++i) {
    const auto& ext_render_message =
        ext_render_per_frame[i].Get<ExternalRenderFrame>();
    EXPECT_EQ(ext_render_message.timestamp_us(), i * 20000);
    EXPECT_EQ(ext_render_message.crop_from_location().x(), 725);
    EXPECT_EQ(ext_render_message.crop_from_location().y(), 0);
    EXPECT_EQ(ext_render_message.crop_from_location().width(), 461);
    EXPECT_EQ(ext_render_message.crop_from_location().height(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().x(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().y(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().width(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().height(), 1124);
  }
}

// Checks external render message with kinematic path solver without video
// input.
TEST(SceneCroppingCalculatorTest, OutputsCropMessageKinematicPathNoVideo) {
  CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kExternalRenderConfigNoVideo, kTargetWidth,
                           kTargetHeight, kKeyFrameWidth, kKeyFrameHeight));
  auto* options = config.mutable_options()->MutableExtension(
      SceneCroppingCalculatorOptions::ext);
  auto* kinematic_options =
      options->mutable_camera_motion_options()->mutable_kinematic_options();
  kinematic_options->set_min_motion_to_reframe(1.2);
  kinematic_options->set_max_velocity(2.0);

  auto runner = absl::make_unique<CalculatorRunner>(config);
  const int num_frames = kSceneSize;
  AddScene(0, num_frames, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, 1, runner->MutableInputs());

  MP_EXPECT_OK(runner->Run());
  const auto& outputs = runner->Outputs();
  const auto& ext_render_per_frame =
      outputs.Tag(kExternalRenderingPerFrameTag).packets;
  EXPECT_EQ(ext_render_per_frame.size(), num_frames);

  for (int i = 0; i < num_frames - 1; ++i) {
    const auto& ext_render_message =
        ext_render_per_frame[i].Get<ExternalRenderFrame>();
    EXPECT_EQ(ext_render_message.timestamp_us(), i * 20000);
    EXPECT_EQ(ext_render_message.crop_from_location().x(), 725);
    EXPECT_EQ(ext_render_message.crop_from_location().y(), 0);
    EXPECT_EQ(ext_render_message.crop_from_location().width(), 461);
    EXPECT_EQ(ext_render_message.crop_from_location().height(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().x(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().y(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().width(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().height(), 1124);
  }
}

// Checks external render message with default poly path solver using
// normalized crops.
TEST(SceneCroppingCalculatorTest, OutputsCropMessagePolyPathNormalized) {
  const CalculatorGraphConfig::Node config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::Substitute(kExternalRenderConfig, kTargetWidth, kTargetHeight));
  auto runner = absl::make_unique<CalculatorRunner>(config);
  const int num_frames = kSceneSize;
  AddScene(0, num_frames, kInputFrameWidth, kInputFrameHeight, kKeyFrameWidth,
           kKeyFrameHeight, 1, runner->MutableInputs());

  MP_EXPECT_OK(runner->Run());
  const auto& outputs = runner->Outputs();
  const auto& ext_render_per_frame =
      outputs.Tag(kExternalRenderingPerFrameTag).packets;
  EXPECT_EQ(ext_render_per_frame.size(), num_frames);

  for (int i = 0; i < num_frames - 1; ++i) {
    const auto& ext_render_message =
        ext_render_per_frame[i].Get<ExternalRenderFrame>();
    EXPECT_EQ(ext_render_message.timestamp_us(), i * 20000);
    EXPECT_EQ(ext_render_message.normalized_crop_from_location().x(),
              725 / static_cast<float>(kInputFrameWidth));
    EXPECT_EQ(ext_render_message.normalized_crop_from_location().y(), 0);
    EXPECT_EQ(ext_render_message.normalized_crop_from_location().width(),
              461 / static_cast<float>(kInputFrameWidth));
    EXPECT_EQ(ext_render_message.normalized_crop_from_location().height(),
              720 / static_cast<float>(kInputFrameHeight));
    EXPECT_EQ(ext_render_message.render_to_location().x(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().y(), 0);
    EXPECT_EQ(ext_render_message.render_to_location().width(), 720);
    EXPECT_EQ(ext_render_message.render_to_location().height(), 1124);
  }
}
}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
