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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/calculators/video/box_tracker_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"
#include "mediapipe/util/tracking/tracking.pb.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif  // defined(__APPLE__)

namespace mediapipe {
namespace {
using ::testing::FloatNear;
using ::testing::Test;

std::string GetTestDir() {
#ifdef __APPLE__
  char path[1024];
  CFURLRef bundle_url = CFBundleCopyBundleURL(CFBundleGetMainBundle());
  CFURLGetFileSystemRepresentation(
      bundle_url, true, reinterpret_cast<UInt8*>(path), sizeof(path));
  CFRelease(bundle_url);
  return ::mediapipe::file::JoinPath(path, "testdata");
#elif defined(__ANDROID__)
  char path[1024];
  getcwd(path, sizeof(path));
  return ::mediapipe::file::JoinPath(path,
                                     "mediapipe/calculators/video/testdata");
#else
  return ::mediapipe::file::JoinPath(
      "./",
      // This should match the path of the output files
      // of the genrule() that generates test model files.
      "mediapipe/calculators/video/testdata");
#endif  // defined(__APPLE__)
}

bool LoadBinaryTestGraph(const std::string& graph_path,
                         CalculatorGraphConfig* config) {
  std::ifstream ifs;
  ifs.open(graph_path.c_str());
  proto_ns::io::IstreamInputStream in_stream(&ifs);
  bool success = config->ParseFromZeroCopyStream(&in_stream);
  ifs.close();
  if (!success) {
    LOG(ERROR) << "could not parse test graph: " << graph_path;
  }
  return success;
}

class TrackingGraphTest : public Test {
 protected:
  TrackingGraphTest() {}

  void SetUp() override {
    test_dir_ = GetTestDir();
    const auto graph_path = file::JoinPath(test_dir_, "tracker.binarypb");
    ASSERT_TRUE(LoadBinaryTestGraph(graph_path, &config_));

    original_image_ = cv::imread(file::JoinPath(test_dir_, "lenna.png"));
    CreateInputFramesFromOriginalImage(kNumImages, kTranslationStep,
                                       &input_frames_packets_);

    const auto& first_input_img = input_frames_packets_[0].Get<ImageFrame>();
    const int img_width = first_input_img.Width();
    const int img_height = first_input_img.Height();
    translation_step_x_ = kTranslationStep / static_cast<float>(img_width);
    translation_step_y_ = kTranslationStep / static_cast<float>(img_height);

    // Creat new configure and packet dump vector to store output.
    mediapipe::CalculatorGraphConfig config_copy = config_;
    mediapipe::tool::AddVectorSink("boxes", &config_copy, &output_packets_);
    mediapipe::tool::AddVectorSink("ra_boxes", &config_copy,
                                   &random_access_results_packets_);

    // Initialize graph
    MP_ASSERT_OK(graph_.Initialize(config_copy));

    const auto parallel_graph_path =
        file::JoinPath(test_dir_, "parallel_tracker.binarypb");
    CalculatorGraphConfig parallel_config;
    ASSERT_TRUE(LoadBinaryTestGraph(parallel_graph_path, &parallel_config));
    mediapipe::tool::AddVectorSink("boxes", &parallel_config, &output_packets_);
    mediapipe::tool::AddVectorSink("ra_boxes", &parallel_config,
                                   &random_access_results_packets_);
    MP_ASSERT_OK(parallel_graph_.Initialize(parallel_config));
  }

  void CreateInputFramesFromOriginalImage(
      int num_images, int translation_step,
      std::vector<Packet>* input_frames_packets);

  void TearDown() override {
    output_packets_.clear();
    random_access_results_packets_.clear();
  }

  std::unique_ptr<TimedBoxProtoList> MakeBoxList(
      const Timestamp& timestamp, const std::vector<bool>& is_quad_tracking,
      const std::vector<bool>& is_pnp_tracking,
      const std::vector<bool>& reacquisition) const;

  void RunGraphWithSidePacketsAndInputs(
      const std::map<std::string, mediapipe::Packet>& side_packets,
      const mediapipe::Packet& start_pos_packet);

  // Utility functions used to judge if a given quad or box is near to the
  // groundtruth location at a given frame.
  // Examine box.reacquisition() field equals to `reacquisition`.
  // `frame` can be float number to account for inter-frame interpolation.
  void ExpectBoxAtFrame(const TimedBoxProto& box, float frame,
                        bool reacquisition);

  // Examine box.aspect_ratio() field equals to `aspect_ratio` if asepct_ratio
  // is positive.
  void ExpectQuadAtFrame(const TimedBoxProto& box, float frame,
                         float aspect_ratio, bool reacquisition);

  // Utility function to judge if two quad are near to each other.
  void ExpectQuadNear(const TimedBoxProto& box1, const TimedBoxProto& box2);

  std::unique_ptr<TimedBoxProtoList> CreateRandomAccessTrackingBoxList(
      const std::vector<Timestamp>& start_timestamps,
      const std::vector<Timestamp>& end_timestamps) const;

  CalculatorGraph graph_;
  CalculatorGraph parallel_graph_;
  CalculatorGraphConfig config_;
  std::string test_dir_;
  cv::Mat original_image_;
  std::vector<Packet> input_frames_packets_;
  std::vector<mediapipe::Packet> output_packets_;
  std::vector<mediapipe::Packet> random_access_results_packets_;
  float translation_step_x_;  // normalized translation step in x direction
  float translation_step_y_;  // normalized translation step in y direction
  static constexpr float kInitialBoxHalfWidthNormalized = 0.25f;
  static constexpr float kInitialBoxHalfHeightNormalized = 0.25f;
  static constexpr float kImageAspectRatio = 1.0f;  // for lenna.png
  static constexpr float kInitialBoxLeft =
      0.5f - kInitialBoxHalfWidthNormalized;
  static constexpr float kInitialBoxRight =
      0.5f + kInitialBoxHalfWidthNormalized;
  static constexpr float kInitialBoxTop =
      0.5f - kInitialBoxHalfHeightNormalized;
  static constexpr float kInitialBoxBottom =
      0.5f + kInitialBoxHalfHeightNormalized;
  static constexpr int kFrameIntervalUs = 30000;
  static constexpr int kNumImages = 8;
  // Each image is shifted to the right and bottom by kTranslationStep
  // pixels compared with the previous image.
  static constexpr int kTranslationStep = 10;
  static constexpr float kEqualityTolerance = 3e-4f;
};

void TrackingGraphTest::ExpectBoxAtFrame(const TimedBoxProto& box, float frame,
                                         bool reacquisition) {
  EXPECT_EQ(box.reacquisition(), reacquisition);
  EXPECT_TRUE(box.has_rotation());
  EXPECT_THAT(box.rotation(), FloatNear(0, kEqualityTolerance));
  EXPECT_THAT(box.left(),
              FloatNear(kInitialBoxLeft - frame * translation_step_x_,
                        kEqualityTolerance));
  EXPECT_THAT(box.top(), FloatNear(kInitialBoxTop - frame * translation_step_y_,
                                   kEqualityTolerance));
  EXPECT_THAT(box.bottom(),
              FloatNear(kInitialBoxBottom - frame * translation_step_y_,
                        kEqualityTolerance));
  EXPECT_THAT(box.right(),
              FloatNear(kInitialBoxRight - frame * translation_step_x_,
                        kEqualityTolerance));
}

void TrackingGraphTest::ExpectQuadAtFrame(const TimedBoxProto& box, float frame,
                                          float aspect_ratio,
                                          bool reacquisition) {
  EXPECT_TRUE(box.has_quad()) << "quad must exist!";
  if (aspect_ratio > 0) {
    EXPECT_TRUE(box.has_aspect_ratio());
    EXPECT_NEAR(box.aspect_ratio(), aspect_ratio, kEqualityTolerance);
  }

  EXPECT_EQ(box.reacquisition(), reacquisition);

  const auto& quad = box.quad();
  EXPECT_EQ(8, quad.vertices_size())
      << "quad has only " << box.quad().vertices_size() << " vertices";
  EXPECT_THAT(quad.vertices(0),
              FloatNear(kInitialBoxLeft - frame * translation_step_x_,
                        kEqualityTolerance));
  EXPECT_THAT(quad.vertices(1),
              FloatNear(kInitialBoxTop - frame * translation_step_y_,
                        kEqualityTolerance));
  EXPECT_THAT(quad.vertices(3),
              FloatNear(kInitialBoxBottom - frame * translation_step_y_,
                        kEqualityTolerance));
  EXPECT_THAT(quad.vertices(4),
              FloatNear(kInitialBoxRight - frame * translation_step_x_,
                        kEqualityTolerance));
}

void TrackingGraphTest::ExpectQuadNear(const TimedBoxProto& box1,
                                       const TimedBoxProto& box2) {
  EXPECT_TRUE(box1.has_quad());
  EXPECT_TRUE(box2.has_quad());
  EXPECT_EQ(8, box1.quad().vertices_size())
      << "quad has only " << box1.quad().vertices_size() << " vertices";
  EXPECT_EQ(8, box2.quad().vertices_size())
      << "quad has only " << box2.quad().vertices_size() << " vertices";
  for (int j = 0; j < box1.quad().vertices_size(); ++j) {
    EXPECT_NEAR(box1.quad().vertices(j), box2.quad().vertices(j),
                kEqualityTolerance);
  }
}

std::unique_ptr<TimedBoxProtoList> TrackingGraphTest::MakeBoxList(
    const Timestamp& timestamp, const std::vector<bool>& is_quad_tracking,
    const std::vector<bool>& is_pnp_tracking,
    const std::vector<bool>& reacquisition) const {
  auto box_list = absl::make_unique<TimedBoxProtoList>();
  int box_id = 0;
  for (int j = 0; j < is_quad_tracking.size(); ++j) {
    TimedBoxProto* box = box_list->add_box();
    if (is_quad_tracking[j]) {
      box->mutable_quad()->add_vertices(kInitialBoxLeft);
      box->mutable_quad()->add_vertices(kInitialBoxTop);
      box->mutable_quad()->add_vertices(kInitialBoxLeft);
      box->mutable_quad()->add_vertices(kInitialBoxBottom);
      box->mutable_quad()->add_vertices(kInitialBoxRight);
      box->mutable_quad()->add_vertices(kInitialBoxBottom);
      box->mutable_quad()->add_vertices(kInitialBoxRight);
      box->mutable_quad()->add_vertices(kInitialBoxTop);

      if (is_pnp_tracking[j]) {
        box->set_aspect_ratio(kImageAspectRatio);
      }
    } else {
      box->set_left(kInitialBoxLeft);
      box->set_right(kInitialBoxRight);
      box->set_top(kInitialBoxTop);
      box->set_bottom(kInitialBoxBottom);
    }
    box->set_id(box_id++);
    box->set_time_msec(timestamp.Value() / 1000);
    box->set_reacquisition(reacquisition[j]);
  }

  return box_list;
}

void TrackingGraphTest::CreateInputFramesFromOriginalImage(
    int num_images, int translation_step,
    std::vector<Packet>* input_frames_packets) {
  const int crop_width = original_image_.cols - num_images * translation_step;
  const int crop_height = original_image_.rows - num_images * translation_step;
  for (int i = 0; i < num_images; ++i) {
    cv::Rect roi(i * translation_step, i * translation_step, crop_width,
                 crop_height);
    cv::Mat cropped_img = cv::Mat(original_image_, roi);
    auto cropped_image_frame = absl::make_unique<ImageFrame>(
        ImageFormat::SRGB, crop_width, crop_height, cropped_img.step[0],
        cropped_img.data, ImageFrame::PixelDataDeleter::kNone);
    Timestamp curr_timestamp = Timestamp(i * kFrameIntervalUs);
    Packet image_packet =
        Adopt(cropped_image_frame.release()).At(curr_timestamp);
    input_frames_packets->push_back(image_packet);
  }
}

void TrackingGraphTest::RunGraphWithSidePacketsAndInputs(
    const std::map<std::string, mediapipe::Packet>& side_packets,
    const mediapipe::Packet& start_pos_packet) {
  // Start running the graph
  MP_EXPECT_OK(graph_.StartRun(side_packets));

  MP_EXPECT_OK(graph_.AddPacketToInputStream("start_pos", start_pos_packet));

  for (auto frame_packet : input_frames_packets_) {
    MP_EXPECT_OK(
        graph_.AddPacketToInputStream("image_cpu_frames", frame_packet));
    MP_EXPECT_OK(graph_.WaitUntilIdle());
  }

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

std::unique_ptr<TimedBoxProtoList>
TrackingGraphTest::CreateRandomAccessTrackingBoxList(
    const std::vector<Timestamp>& start_timestamps,
    const std::vector<Timestamp>& end_timestamps) const {
  CHECK_EQ(start_timestamps.size(), end_timestamps.size());
  auto ra_boxes = absl::make_unique<TimedBoxProtoList>();
  for (int i = 0; i < start_timestamps.size(); ++i) {
    auto start_box_list =
        MakeBoxList(start_timestamps[i], std::vector<bool>{true},
                    std::vector<bool>{true}, std::vector<bool>{false});
    auto end_box_list =
        MakeBoxList(end_timestamps[i], std::vector<bool>{true},
                    std::vector<bool>{true}, std::vector<bool>{false});
    *(ra_boxes->add_box()) = (*start_box_list).box(0);
    *(ra_boxes->add_box()) = (*end_box_list).box(0);
  }
  return ra_boxes;
}

TEST_F(TrackingGraphTest, BasicBoxTrackingSanityCheck) {
  // Create input side packets.
  std::map<std::string, mediapipe::Packet> side_packets;
  side_packets.insert(std::make_pair("analysis_downsample_factor",
                                     mediapipe::MakePacket<float>(1.0f)));
  side_packets.insert(std::make_pair(
      "calculator_options",
      mediapipe::MakePacket<CalculatorOptions>(CalculatorOptions())));

  // Run the graph with input side packets, start_pos, and input image frames.
  Timestamp start_box_time = input_frames_packets_[0].Timestamp();
  // is_quad_tracking is used to indicate whether to track quad for each
  // individual box.
  std::vector<bool> is_quad_tracking{false};
  // is_pnp_tracking is used to indicate whether to use perspective transform to
  // track quad.
  std::vector<bool> is_pnp_tracking{false};
  // is_reacquisition is used to indicate whether to enable reacquisition for
  // the box.
  std::vector<bool> is_reacquisition{false};
  auto start_box_list = MakeBoxList(start_box_time, is_quad_tracking,
                                    is_pnp_tracking, is_reacquisition);
  Packet start_pos_packet = Adopt(start_box_list.release()).At(start_box_time);
  RunGraphWithSidePacketsAndInputs(side_packets, start_pos_packet);

  EXPECT_EQ(input_frames_packets_.size(), output_packets_.size());

  for (int i = 0; i < output_packets_.size(); ++i) {
    const TimedBoxProtoList& boxes =
        output_packets_[i].Get<TimedBoxProtoList>();
    EXPECT_EQ(is_quad_tracking.size(), boxes.box_size());
    ExpectBoxAtFrame(boxes.box(0), i, false);
  }
}

TEST_F(TrackingGraphTest, BasicQuadTrackingSanityCheck) {
  // Create input side packets.
  std::map<std::string, mediapipe::Packet> side_packets;
  side_packets.insert(std::make_pair("analysis_downsample_factor",
                                     mediapipe::MakePacket<float>(1.0f)));
  CalculatorOptions calculator_options;
  calculator_options.MutableExtension(BoxTrackerCalculatorOptions::ext)
      ->mutable_tracker_options()
      ->mutable_track_step_options()
      ->set_tracking_degrees(
          TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE);
  side_packets.insert(std::make_pair(
      "calculator_options",
      mediapipe::MakePacket<CalculatorOptions>(calculator_options)));

  Timestamp start_box_time = input_frames_packets_[0].Timestamp();
  // Box id 0 use quad tracking with 8DoF homography transform.
  // Box id 1 use quad tracking with 6DoF perspective transform.
  // Box id 2 use box tracking with 4DoF similarity transform.
  std::vector<bool> is_quad_tracking{true, true, false};
  std::vector<bool> is_pnp_tracking{false, true, false};
  std::vector<bool> is_reacquisition{true, false, true};
  auto start_box_list = MakeBoxList(start_box_time, is_quad_tracking,
                                    is_pnp_tracking, is_reacquisition);
  Packet start_pos_packet = Adopt(start_box_list.release()).At(start_box_time);
  RunGraphWithSidePacketsAndInputs(side_packets, start_pos_packet);

  EXPECT_EQ(input_frames_packets_.size(), output_packets_.size());
  for (int i = 0; i < output_packets_.size(); ++i) {
    const TimedBoxProtoList& boxes =
        output_packets_[i].Get<TimedBoxProtoList>();
    EXPECT_EQ(is_quad_tracking.size(), boxes.box_size());
    for (int j = 0; j < boxes.box_size(); ++j) {
      const TimedBoxProto& box = boxes.box(j);
      if (is_quad_tracking[box.id()]) {
        ExpectQuadAtFrame(box, i,
                          is_pnp_tracking[box.id()] ? kImageAspectRatio : -1.0f,
                          is_reacquisition[box.id()]);
      } else {
        ExpectBoxAtFrame(box, i, is_reacquisition[box.id()]);
      }
    }
  }
}

TEST_F(TrackingGraphTest, TestRandomAccessTrackingResults) {
  // Create input side packets.
  std::map<std::string, mediapipe::Packet> side_packets;
  side_packets.insert(std::make_pair("analysis_downsample_factor",
                                     mediapipe::MakePacket<float>(1.0f)));
  CalculatorOptions calculator_options;
  calculator_options.MutableExtension(BoxTrackerCalculatorOptions::ext)
      ->mutable_tracker_options()
      ->mutable_track_step_options()
      ->set_tracking_degrees(
          TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE);
  side_packets.insert(std::make_pair(
      "calculator_options",
      mediapipe::MakePacket<CalculatorOptions>(calculator_options)));

  ASSERT_GT(input_frames_packets_.size(), 2);  // at least 3 frames
  ASSERT_TRUE(input_frames_packets_[2].Timestamp() -
                  input_frames_packets_[1].Timestamp() >
              TimestampDiff(1000));

  constexpr int start_frame = 0;
  Timestamp start_box_time = input_frames_packets_[start_frame].Timestamp();
  auto start_box_list =
      MakeBoxList(start_box_time, std::vector<bool>{true},
                  std::vector<bool>{true}, std::vector<bool>{false});
  constexpr int end_frame = 2;
  Timestamp end_box_time = input_frames_packets_[end_frame].Timestamp();

  // Also test reverse random access tracking.
  // This offset of 1ms is simulating the case where the start query timestamp
  // to be not any existing frame timestamp. In reality, it's highly encouraged
  // to have the start query timestamp be aligned with frame timestamp.
  constexpr int reverse_start_frame = 1;
  Timestamp reverse_start_box_time =
      input_frames_packets_[reverse_start_frame].Timestamp() + 1000;

  auto ra_boxes = CreateRandomAccessTrackingBoxList(
      {start_box_time, reverse_start_box_time}, {end_box_time, start_box_time});

  Packet ra_packet = Adopt(ra_boxes.release()).At(start_box_time);
  Packet start_packet = Adopt(start_box_list.release()).At(start_box_time);

  // Start running the ordinary graph, verify random access produce same result
  // as normal tracking.
  MP_EXPECT_OK(graph_.StartRun(side_packets));
  MP_EXPECT_OK(graph_.AddPacketToInputStream("start_pos", start_packet));
  for (auto frame_packet : input_frames_packets_) {
    MP_EXPECT_OK(
        graph_.AddPacketToInputStream("image_cpu_frames", frame_packet));
    Packet track_time_packet = Adopt(new int(0)).At(frame_packet.Timestamp());
    MP_EXPECT_OK(
        graph_.AddPacketToInputStream("track_time", track_time_packet));
    MP_EXPECT_OK(graph_.WaitUntilIdle());
  }
  MP_EXPECT_OK(graph_.AddPacketToInputStream("ra_track", ra_packet));
  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());

  EXPECT_EQ(input_frames_packets_.size(), output_packets_.size());
  const TimedBoxProtoList tracking_result =
      output_packets_[end_frame].Get<TimedBoxProtoList>();
  EXPECT_EQ(1, tracking_result.box_size());

  // Should have 1 random access packet.
  EXPECT_EQ(1, random_access_results_packets_.size());
  const TimedBoxProtoList& ra_result =
      random_access_results_packets_[0].Get<TimedBoxProtoList>();
  // Two box tracking results. One for comparison with normal tracking. The
  // other for reverse random access tracking.
  EXPECT_EQ(2, ra_result.box_size());

  // Check if randan access tracking has same result with normal tracking.
  ExpectQuadNear(tracking_result.box(0), ra_result.box(0));
  ExpectQuadAtFrame(ra_result.box(0), end_frame - start_frame,
                    kImageAspectRatio, false);
  ExpectQuadAtFrame(ra_result.box(1), start_frame - reverse_start_frame - 1,
                    kImageAspectRatio, false);

  // Clear output and ra result packet vector before test parallel graph.
  TearDown();

  // Start running the parallel graph, verify random access produce same result
  // as normal tracking.
  MP_EXPECT_OK(parallel_graph_.StartRun(side_packets));
  MP_EXPECT_OK(
      parallel_graph_.AddPacketToInputStream("start_pos", start_packet));
  for (auto frame_packet : input_frames_packets_) {
    MP_EXPECT_OK(parallel_graph_.AddPacketToInputStream("image_cpu_frames",
                                                        frame_packet));
    MP_EXPECT_OK(parallel_graph_.WaitUntilIdle());
  }
  MP_EXPECT_OK(parallel_graph_.AddPacketToInputStream("ra_track", ra_packet));
  MP_EXPECT_OK(parallel_graph_.CloseAllInputStreams());
  MP_EXPECT_OK(parallel_graph_.WaitUntilDone());

  EXPECT_EQ(input_frames_packets_.size(), output_packets_.size());
  const TimedBoxProtoList parallel_tracking_result =
      output_packets_[end_frame].Get<TimedBoxProtoList>();
  EXPECT_EQ(1, parallel_tracking_result.box_size());

  // should have only 1 random access
  EXPECT_EQ(1, random_access_results_packets_.size());
  const TimedBoxProtoList& parallel_ra_result =
      random_access_results_packets_[0].Get<TimedBoxProtoList>();
  EXPECT_EQ(2, parallel_ra_result.box_size());

  // Check if randan access tracking has same result with normal tracking.
  ExpectQuadNear(parallel_tracking_result.box(0), parallel_ra_result.box(0));
  ExpectQuadAtFrame(parallel_ra_result.box(0), end_frame - start_frame,
                    kImageAspectRatio, false);
  ExpectQuadAtFrame(parallel_ra_result.box(1),
                    start_frame - reverse_start_frame - 1, kImageAspectRatio,
                    false);
}

// Tests what happens when random access request timestamps are
// outside of cache.
TEST_F(TrackingGraphTest, TestRandomAccessTrackingTimestamps) {
  // Create input side packets.
  std::map<std::string, mediapipe::Packet> side_packets;
  side_packets.insert(std::make_pair("analysis_downsample_factor",
                                     mediapipe::MakePacket<float>(1.0f)));
  CalculatorOptions calculator_options;
  calculator_options.MutableExtension(BoxTrackerCalculatorOptions::ext)
      ->mutable_tracker_options()
      ->mutable_track_step_options()
      ->set_tracking_degrees(
          TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE);
  // We intentionally don't cache all frames, to see what happens when
  // random access tracking request time falls outside cache range.
  calculator_options.MutableExtension(BoxTrackerCalculatorOptions::ext)
      ->set_streaming_track_data_cache_size(input_frames_packets_.size() - 1);
  side_packets.insert(std::make_pair(
      "calculator_options",
      mediapipe::MakePacket<CalculatorOptions>(calculator_options)));

  // Set up random access boxes
  const int num_frames = input_frames_packets_.size();
  const int64 usec_in_sec = 1000000;
  std::vector<Timestamp> start_timestamps{
      input_frames_packets_[0].Timestamp() - usec_in_sec,  // forward
      input_frames_packets_[0].Timestamp(),                // forward
      input_frames_packets_[1].Timestamp(),                // forward
      input_frames_packets_[num_frames - 1].Timestamp() + usec_in_sec,  // fwd
      input_frames_packets_[0].Timestamp(),               // backward
      input_frames_packets_[num_frames - 1].Timestamp(),  // backward
      input_frames_packets_[num_frames - 1].Timestamp(),  // backward
      input_frames_packets_[num_frames - 1].Timestamp() + usec_in_sec  // back
  };
  std::vector<Timestamp> end_timestamps{
      input_frames_packets_[num_frames - 1].Timestamp(),
      input_frames_packets_[num_frames - 1].Timestamp(),
      input_frames_packets_[num_frames - 1].Timestamp() + usec_in_sec,
      input_frames_packets_[num_frames - 1].Timestamp() + 2 * usec_in_sec,
      input_frames_packets_[0].Timestamp() - usec_in_sec,
      input_frames_packets_[0].Timestamp(),
      input_frames_packets_[0].Timestamp() - usec_in_sec,
      input_frames_packets_[1].Timestamp()};
  auto ra_boxes =
      CreateRandomAccessTrackingBoxList(start_timestamps, end_timestamps);
  Packet ra_packet =
      Adopt(ra_boxes.release()).At(input_frames_packets_[0].Timestamp());

  // Run the graph and check if the outside-cache request have no results.
  // Start running the parallel graph, verify random access produce same result
  // as normal tracking.
  MP_EXPECT_OK(parallel_graph_.StartRun(side_packets));
  for (auto frame_packet : input_frames_packets_) {
    MP_EXPECT_OK(parallel_graph_.AddPacketToInputStream("image_cpu_frames",
                                                        frame_packet));
    MP_EXPECT_OK(parallel_graph_.WaitUntilIdle());
  }
  MP_EXPECT_OK(parallel_graph_.AddPacketToInputStream("ra_track", ra_packet));
  MP_EXPECT_OK(parallel_graph_.CloseAllInputStreams());
  MP_EXPECT_OK(parallel_graph_.WaitUntilDone());

  // should have 1 random access packet with 0 result boxes
  EXPECT_EQ(1, random_access_results_packets_.size());
  const auto& ra_returned_boxes =
      random_access_results_packets_[0].Get<TimedBoxProtoList>();
  const int num_returned_ra_boxes = ra_returned_boxes.box_size();
  EXPECT_EQ(0, num_returned_ra_boxes);
}

TEST_F(TrackingGraphTest, TestTransitionFramesForReacquisition) {
  // Create input side packets.
  std::map<std::string, mediapipe::Packet> side_packets;
  side_packets.insert(std::make_pair("analysis_downsample_factor",
                                     mediapipe::MakePacket<float>(1.0f)));
  CalculatorOptions calculator_options;
  calculator_options.MutableExtension(BoxTrackerCalculatorOptions::ext)
      ->mutable_tracker_options()
      ->mutable_track_step_options()
      ->set_tracking_degrees(
          TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE);
  constexpr int kTransitionFrames = 3;
  calculator_options.MutableExtension(BoxTrackerCalculatorOptions::ext)
      ->set_start_pos_transition_frames(kTransitionFrames);

  side_packets.insert(std::make_pair(
      "calculator_options",
      mediapipe::MakePacket<CalculatorOptions>(calculator_options)));

  Timestamp start_box_time = input_frames_packets_[0].Timestamp();
  // Box id 0 use quad tracking with 8DoF homography transform.
  // Box id 1 use quad tracking with 6DoF perspective transform.
  // Box id 2 use box tracking with 4DoF similarity transform.
  std::vector<bool> is_quad_tracking{true, true, false};
  std::vector<bool> is_pnp_tracking{false, true, false};
  std::vector<bool> is_reacquisition{true, true, true};
  auto start_box_list = MakeBoxList(start_box_time, is_quad_tracking,
                                    is_pnp_tracking, is_reacquisition);
  Packet start_pos_packet = Adopt(start_box_list.release()).At(start_box_time);

  // Setting box pos restart from initial position (frame 0's position).
  constexpr int kRestartFrame = 3;
  Timestamp restart_box_time = input_frames_packets_[kRestartFrame].Timestamp();
  auto restart_box_list = MakeBoxList(restart_box_time, is_quad_tracking,
                                      is_pnp_tracking, is_reacquisition);
  Packet restart_pos_packet =
      Adopt(restart_box_list.release()).At(restart_box_time);
  MP_EXPECT_OK(graph_.StartRun(side_packets));
  MP_EXPECT_OK(graph_.AddPacketToInputStream("start_pos", start_pos_packet));

  for (int j = 0; j < input_frames_packets_.size(); ++j) {
    // Add TRACK_TIME stream queries in between 2 frames.
    if (j > 0) {
      Timestamp track_time = Timestamp((j - 0.5f) * kFrameIntervalUs);
      LOG(INFO) << track_time.Value();
      Packet track_time_packet = Adopt(new Timestamp).At(track_time);
      MP_EXPECT_OK(
          graph_.AddPacketToInputStream("track_time", track_time_packet));
    }

    MP_EXPECT_OK(graph_.AddPacketToInputStream("image_cpu_frames",
                                               input_frames_packets_[j]));
    Packet track_time_packet =
        Adopt(new int(0)).At(input_frames_packets_[j].Timestamp());
    MP_EXPECT_OK(
        graph_.AddPacketToInputStream("track_time", track_time_packet));
    MP_EXPECT_OK(graph_.WaitUntilIdle());

    if (j == kRestartFrame) {
      MP_EXPECT_OK(
          graph_.AddPacketToInputStream("restart_pos", restart_pos_packet));
    }
  }

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());

  EXPECT_EQ(input_frames_packets_.size() * 2 - 1, output_packets_.size());
  for (int i = 0; i < output_packets_.size(); ++i) {
    const TimedBoxProtoList& boxes =
        output_packets_[i].Get<TimedBoxProtoList>();
    EXPECT_EQ(is_quad_tracking.size(), boxes.box_size());
    float frame_id = i / 2.0f;
    float expected_frame_id;
    if (frame_id <= kRestartFrame) {
      // before transition
      expected_frame_id = frame_id;
    } else {
      float transition_frames = frame_id - kRestartFrame;
      if (transition_frames <= kTransitionFrames) {
        // transitioning.
        expected_frame_id =
            kRestartFrame -
            transition_frames / kTransitionFrames * kRestartFrame +
            transition_frames;
      } else {
        // after transition.
        expected_frame_id = transition_frames;
      }
    }

    for (int j = 0; j < boxes.box_size(); ++j) {
      const TimedBoxProto& box = boxes.box(j);
      if (is_quad_tracking[box.id()]) {
        ExpectQuadAtFrame(box, expected_frame_id,
                          is_pnp_tracking[box.id()] ? kImageAspectRatio : -1.0f,
                          is_reacquisition[box.id()]);
      } else {
        ExpectBoxAtFrame(box, expected_frame_id, is_reacquisition[box.id()]);
      }
    }
  }
}

// TODO: Add test for reacquisition.

}  // namespace
}  // namespace mediapipe
