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

#include "mediapipe/util/tracking/region_flow_computation.h"

#include <math.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "absl/time/clock.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/tracking/region_flow.h"
#include "mediapipe/util/tracking/region_flow.pb.h"

// To ensure that the selected thresholds are robust, it is recommend
// to run this test mutiple times with time seed, if changes are made.
DEFINE_bool(time_seed, false, "Activate to test thresholds");

namespace mediapipe {
namespace {

using RandomEngine = std::mt19937_64;

struct FlowDirectionParam {
  TrackingOptions::FlowDirection internal_direction;
  TrackingOptions::FlowDirection output_direction;
};

class RegionFlowComputationTest
    : public ::testing::TestWithParam<FlowDirectionParam> {
 protected:
  virtual void SetUp() {
    // Setup tracking direction options based on testing parameter.
    const auto& param = GetParam();
    auto* tracking_options = base_options_.mutable_tracking_options();
    tracking_options->set_internal_tracking_direction(param.internal_direction);
    tracking_options->set_output_flow_direction(param.output_direction);

    // Load bee image.
    data_dir_ = file::JoinPath("./", "/mediapipe/util/tracking/testdata/");

    std::string png_data;
    MEDIAPIPE_CHECK_OK(
        file::GetContents(data_dir_ + "stabilize_test.png", &png_data));
    std::vector<char> buffer(png_data.begin(), png_data.end());
    original_frame_ = cv::imdecode(cv::Mat(buffer), 1);
    ASSERT_FALSE(original_frame_.empty());
    ASSERT_EQ(original_frame_.type(), CV_8UC3);
  }

  // Creates a movie in the specified format by displacing original_frame_ to
  // random positions.
  void MakeMovie(int frames, RegionFlowComputationOptions::ImageFormat format,
                 std::vector<cv::Mat>* movie,
                 std::vector<Vector2_f>* positions);

  // Outputs allocated resized input frame.
  void GetResizedFrame(int width, int height, cv::Mat* result) const;

  // Runs frame pair test using RGB, RGBA or grayscale input.
  void RunFramePairTest(RegionFlowComputationOptions::ImageFormat format);

 protected:
  RegionFlowComputationOptions base_options_;

 private:
  std::string data_dir_;
  cv::Mat original_frame_;
};

std::vector<FlowDirectionParam> FlowDirectionCombinations() {
  return {{TrackingOptions::FORWARD, TrackingOptions::FORWARD},
          {TrackingOptions::FORWARD, TrackingOptions::BACKWARD},
          {TrackingOptions::BACKWARD, TrackingOptions::FORWARD},
          {TrackingOptions::BACKWARD, TrackingOptions::BACKWARD}};
}

INSTANTIATE_TEST_SUITE_P(FlowDirection, RegionFlowComputationTest,
                         ::testing::ValuesIn(FlowDirectionCombinations()));

void RegionFlowComputationTest::MakeMovie(
    int num_frames, RegionFlowComputationOptions::ImageFormat format,
    std::vector<cv::Mat>* movie, std::vector<Vector2_f>* positions) {
  CHECK(positions != nullptr);
  CHECK(movie != nullptr);

  const int border = 40;
  int frame_width = original_frame_.cols - 2 * border;
  int frame_height = original_frame_.rows - 2 * border;
  ASSERT_GT(frame_width, 0);
  ASSERT_GT(frame_height, 0);

  // First generate random positions.
  int seed = 900913;  // google.
  if (FLAGS_time_seed) {
    seed = ToUnixMillis(absl::Now()) % (1 << 16);
    LOG(INFO) << "Using time seed: " << seed;
  }

  RandomEngine random(seed);
  std::uniform_int_distribution<> uniform_dist(-10, 10);
  positions->resize(num_frames);
  (*positions)[0] = Vector2_f(border, border);
  for (int f = 1; f < num_frames; ++f) {
    Vector2_f pos = (*positions)[f - 1] +
                    Vector2_f(uniform_dist(random), uniform_dist(random));

    // Clamp to valid positions.
    pos.x(std::max<int>(0, pos.x()));
    pos.y(std::max<int>(0, pos.y()));
    pos.x(std::min<int>(2 * border, pos.x()));
    pos.y(std::min<int>(2 * border, pos.y()));

    (*positions)[f] = pos;
  }

  // Create movie by copying.
  movie->resize(num_frames);
  cv::Mat original_frame = original_frame_;

  auto convert = [&](int channel_format, int conversion_code) {
    original_frame.create(original_frame_.rows, original_frame_.cols,
                          channel_format);
    cv::cvtColor(original_frame_, original_frame, conversion_code);
  };

  switch (format) {
    case RegionFlowComputationOptions::FORMAT_RGB:
      break;

    case RegionFlowComputationOptions::FORMAT_BGR:
      convert(CV_8UC3, cv::COLOR_RGB2BGR);
      break;

    case RegionFlowComputationOptions::FORMAT_GRAYSCALE:
      convert(CV_8UC1, cv::COLOR_RGB2GRAY);
      break;

    case RegionFlowComputationOptions::FORMAT_RGBA:
      convert(CV_8UC4, cv::COLOR_RGB2RGBA);
      break;

    case RegionFlowComputationOptions::FORMAT_BGRA:
      convert(CV_8UC4, cv::COLOR_RGB2BGRA);
      break;
  }
  for (int f = 0; f < num_frames; ++f) {
    (*movie)[f].create(frame_height, frame_width, original_frame.type());
    const auto& pos = (*positions)[f];
    cv::Mat tmp_view(original_frame, cv::Range(pos.y(), pos.y() + frame_height),
                     cv::Range(pos.x(), pos.x() + frame_width));
    tmp_view.copyTo((*movie)[f]);
  }
}

void RegionFlowComputationTest::GetResizedFrame(int width, int height,
                                                cv::Mat* result) const {
  CHECK(result != nullptr);
  cv::resize(original_frame_, *result, cv::Size(width, height));
}

void RegionFlowComputationTest::RunFramePairTest(
    RegionFlowComputationOptions::ImageFormat format) {
  std::vector<cv::Mat> movie;
  std::vector<Vector2_f> positions;
  const int num_frames = 10;
  MakeMovie(num_frames, format, &movie, &positions);

  const int frame_width = movie[0].cols;
  const int frame_height = movie[0].rows;

  base_options_.set_image_format(format);

  RegionFlowComputation flow_computation(base_options_, frame_width,
                                         frame_height);

  for (int i = 0; i < num_frames; ++i) {
    flow_computation.AddImage(movie[i], 0);

    if (i > 0) {
      float inliers = 0;
      std::unique_ptr<RegionFlowFrame> region_flow_frame(
          flow_computation.RetrieveRegionFlow());
      // Get flow vector based on actual motion applied to frames. Direction
      // based on output flow direction.
      Vector2_f flow_vector;
      switch (base_options_.tracking_options().output_flow_direction()) {
        case TrackingOptions::BACKWARD:
          flow_vector = positions[i] - positions[i - 1];
          break;
        case TrackingOptions::FORWARD:
          flow_vector = positions[i - 1] - positions[i];
          break;
        case TrackingOptions::CONSECUTIVELY:
          FAIL();
          break;
      }

      // We expect all flow vectors to be similar to flow_vector.
      for (const auto& region_flow : region_flow_frame->region_flow()) {
        for (const auto& feature : region_flow.feature()) {
          // Half a pixel error is very reasonable.
          if (fabs(flow_vector.x() - FeatureFlow(feature).x()) < 0.5f &&
              fabs(flow_vector.y() - FeatureFlow(feature).y()) < 0.5f) {
            ++inliers;
          }
        }
      }
      // 95% of all features should be tracked reliably.
      EXPECT_GE(inliers / region_flow_frame->num_total_features(), 0.95f);
    }
  }
}

TEST_P(RegionFlowComputationTest, FramePairTest) {
  // The output flow direction should be either forward or backward.
  EXPECT_NE(base_options_.tracking_options().output_flow_direction(),
            TrackingOptions::CONSECUTIVELY);
  // Test on grayscale input.
  RunFramePairTest(RegionFlowComputationOptions::FORMAT_GRAYSCALE);
  // Test on RGB input.
  RunFramePairTest(RegionFlowComputationOptions::FORMAT_RGB);
  // Test on BGR input.
  RunFramePairTest(RegionFlowComputationOptions::FORMAT_BGR);
  // Test on RGBA input.
  RunFramePairTest(RegionFlowComputationOptions::FORMAT_RGBA);
  // Test on BGRA input.
  RunFramePairTest(RegionFlowComputationOptions::FORMAT_BGRA);
}

TEST_P(RegionFlowComputationTest, ResolutionTests) {
  // Test all kinds of resolutions (disregard resulting flow).
  // Square test, synthetic tracks.
  for (int dim = 1; dim <= 50; ++dim) {
    RegionFlowComputationOptions options = base_options_;
    // Force synthetic features, as images are too small to track anything.
    options.set_use_synthetic_zero_motion_tracks_all_frames(true);
    RegionFlowComputation flow_computation(options, dim, dim);

    cv::Mat input_frame;
    GetResizedFrame(dim, dim, &input_frame);
    // Add frame several times.
    for (int i = 0; i < 5; ++i) {
      flow_computation.AddImage(input_frame, 0);

      // Don't care about the result here, simply test for segfaults.
      delete flow_computation.RetrieveRegionFlow();
    }
  }

  // Larger frames with tracking.
  for (int dim = 50; dim <= 100; ++dim) {
    RegionFlowComputation flow_computation(base_options_, dim, dim);

    cv::Mat input_frame;
    GetResizedFrame(dim, dim, &input_frame);
    for (int i = 0; i < 5; ++i) {
      flow_computation.AddImage(input_frame, 0);
      delete flow_computation.RetrieveRegionFlow();
    }
  }

  // Different aspect ratios, first frame synthetic only.
  for (int y = 1; y <= 50; y += 3) {
    for (int x = 1; x <= 100; x += 7) {
      RegionFlowComputationOptions options = base_options_;
      options.set_use_synthetic_zero_motion_tracks_first_frame(true);
      RegionFlowComputation flow_computation(options, x, y);

      cv::Mat input_frame;
      GetResizedFrame(x, y, &input_frame);
      for (int i = 0; i < 5; ++i) {
        flow_computation.AddImage(input_frame, 0);
        delete flow_computation.RetrieveRegionFlow();
      }
    }
  }
}

}  // namespace
}  // namespace mediapipe
