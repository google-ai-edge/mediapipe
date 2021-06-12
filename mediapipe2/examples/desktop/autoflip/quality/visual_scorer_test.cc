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

#include "mediapipe/examples/desktop/autoflip/quality/visual_scorer.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {
namespace {

TEST(VisualScorerTest, ScoresArea) {
  cv::Mat image_mat(200, 200, CV_8UC3);
  SalientRegion region = ParseTextProtoOrDie<SalientRegion>(
      R"pb(location { x: 10 y: 10 width: 100 height: 100 })pb");

  VisualScorerOptions options = ParseTextProtoOrDie<VisualScorerOptions>(
      R"pb(area_weight: 1.0 sharpness_weight: 0 colorfulness_weight: 0)pb");
  VisualScorer scorer(options);
  float score = 0.0;
  MP_EXPECT_OK(scorer.CalculateScore(image_mat, region, &score));
  EXPECT_EQ(0.25, score);  // (100 * 100) / (200 * 200).
}

TEST(VisualScorerTest, ScoresSharpness) {
  SalientRegion region = ParseTextProtoOrDie<SalientRegion>(
      R"pb(location { x: 10 y: 10 width: 100 height: 100 })pb");

  VisualScorerOptions options = ParseTextProtoOrDie<VisualScorerOptions>(
      R"pb(area_weight: 0 sharpness_weight: 1.0 colorfulness_weight: 0)pb");
  VisualScorer scorer(options);

  // Compute the score of an empty image and an image with a rectangle.
  cv::Mat image_mat(200, 200, CV_8UC3);
  image_mat.setTo(cv::Scalar(0, 0, 0));
  float score_rect = 0;
  auto status = scorer.CalculateScore(image_mat, region, &score_rect);
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(VisualScorerTest, ScoresColorfulness) {
  SalientRegion region = ParseTextProtoOrDie<SalientRegion>(
      R"pb(location { x: 10 y: 10 width: 50 height: 150 })pb");

  VisualScorerOptions options = ParseTextProtoOrDie<VisualScorerOptions>(
      R"pb(area_weight: 0 sharpness_weight: 0 colorfulness_weight: 1.0)pb");
  VisualScorer scorer(options);

  // Compute the scores of images with 1, 2 and 3 colors.
  cv::Mat image_mat(200, 200, CV_8UC3);
  image_mat.setTo(cv::Scalar(0, 0, 255));
  float score_1c = 0, score_2c = 0, score_3c = 0;
  MP_EXPECT_OK(scorer.CalculateScore(image_mat, region, &score_1c));
  image_mat(cv::Rect(30, 30, 20, 20)).setTo(cv::Scalar(128, 0, 0));
  MP_EXPECT_OK(scorer.CalculateScore(image_mat, region, &score_2c));
  image_mat(cv::Rect(50, 50, 20, 20)).setTo(cv::Scalar(255, 128, 0));
  MP_EXPECT_OK(scorer.CalculateScore(image_mat, region, &score_3c));
  // Images with more colors should have a higher score.
  EXPECT_LT(score_1c, score_2c);
  EXPECT_LT(score_2c, score_3c);
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
