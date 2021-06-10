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

#include <math.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

// Weight threshold for computing a value.
constexpr float kEpsilon = 0.0001;

namespace mediapipe {
namespace autoflip {
namespace {

// Crop the given rectangle so that it fits in the given 2D matrix.
void CropRectToMat(const cv::Mat& image, cv::Rect* rect) {
  int x = std::min(std::max(rect->x, 0), image.cols);
  int y = std::min(std::max(rect->y, 0), image.rows);
  int w = std::min(std::max(rect->x + rect->width, 0), image.cols) - x;
  int h = std::min(std::max(rect->y + rect->height, 0), image.rows) - y;
  *rect = cv::Rect(x, y, w, h);
}

}  // namespace

VisualScorer::VisualScorer(const VisualScorerOptions& options)
    : options_(options) {}

absl::Status VisualScorer::CalculateScore(const cv::Mat& image,
                                          const SalientRegion& region,
                                          float* score) const {
  const float weight_sum = options_.area_weight() +
                           options_.sharpness_weight() +
                           options_.colorfulness_weight();

  // Crop the region to fit in the image.
  cv::Rect region_rect;
  if (region.has_location()) {
    region_rect =
        cv::Rect(region.location().x(), region.location().y(),
                 region.location().width(), region.location().height());
  } else if (region.has_location_normalized()) {
    region_rect = cv::Rect(region.location_normalized().x() * image.cols,
                           region.location_normalized().y() * image.rows,
                           region.location_normalized().width() * image.cols,
                           region.location_normalized().height() * image.rows);
  } else {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Unset region location.";
  }

  CropRectToMat(image, &region_rect);
  if (region_rect.area() == 0) {
    *score = 0;
    return absl::OkStatus();
  }

  // Compute a score based on area covered by this region.
  const float area_score =
      options_.area_weight() * region_rect.area() / (image.cols * image.rows);

  // Convert the visible region to cv::Mat.
  cv::Mat image_region_mat = image(region_rect);

  // Compute a score from sharpness.

  float sharpness_score_result = 0.0;
  if (options_.sharpness_weight() > kEpsilon) {
    // TODO: implement a sharpness score or remove this code block.
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "sharpness scorer is not yet implemented, please set weight to "
              "0.0";
  }
  const float sharpness_score =
      options_.sharpness_weight() * sharpness_score_result;

  // Compute a colorfulness score.
  float colorfulness_score = 0;
  if (options_.colorfulness_weight() > kEpsilon) {
    MP_RETURN_IF_ERROR(
        CalculateColorfulness(image_region_mat, &colorfulness_score));
    colorfulness_score *= options_.colorfulness_weight();
  }

  *score = (area_score + sharpness_score + colorfulness_score) / weight_sum;
  if (*score > 1.0f || *score < 0.0f) {
    LOG(WARNING) << "Score of region outside expected range: " << *score;
  }
  return absl::OkStatus();
}

absl::Status VisualScorer::CalculateColorfulness(const cv::Mat& image,
                                                 float* colorfulness) const {
  // Convert the image to HSV.
  cv::Mat image_hsv;
  cv::cvtColor(image, image_hsv, CV_RGB2HSV);

  // Mask out pixels that are too dark or too bright.
  cv::Mat mask(image.rows, image.cols, CV_8UC1);
  bool empty_mask = true;
  for (int x = 0; x < image.cols; ++x) {
    for (int y = 0; y < image.rows; ++y) {
      const cv::Vec3b& pixel = image.at<cv::Vec3b>(x, y);
      const bool is_usable =
          (std::min(pixel.val[0], std::min(pixel.val[1], pixel.val[2])) < 250 &&
           std::max(pixel.val[0], std::max(pixel.val[1], pixel.val[2])) > 5);
      mask.at<unsigned char>(y, x) = is_usable ? 255 : 0;
      if (is_usable) empty_mask = false;
    }
  }

  // If the mask is empty, return.
  if (empty_mask) {
    *colorfulness = 0;
    return absl::OkStatus();
  }

  // Generate a 2D histogram (hue/saturation).
  cv::MatND hs_histogram;
  const int kHueBins = 10, kSaturationBins = 8;
  const int kHistogramChannels[] = {0, 1};
  const int kHistogramBinNum[] = {kHueBins, kSaturationBins};
  const float kHueRange[] = {0, 180};
  const float kSaturationRange[] = {0, 256};
  const float* kHistogramRange[] = {kHueRange, kSaturationRange};
  cv::calcHist(&image_hsv, 1, kHistogramChannels, mask, hs_histogram,
               2 /* histogram dims */, kHistogramBinNum, kHistogramRange,
               true /* uniform */, false /* accumulate */);

  // Convert to a hue histogram and weigh saturated pixels more.
  std::vector<float> hue_histogram(kHueBins, 0.0f);
  float hue_sum = 0.0f;
  for (int bin_s = 0; bin_s < kSaturationBins; ++bin_s) {
    const float weight = std::pow(2.0f, bin_s);
    for (int bin_h = 0; bin_h < kHueBins; ++bin_h) {
      float value = hs_histogram.at<float>(bin_h, bin_s) * weight;
      hue_histogram[bin_h] += value;
      hue_sum += value;
    }
  }
  if (hue_sum == 0.0f) {
    *colorfulness = 0;
    return absl::OkStatus();
  }

  // Compute the histogram entropy.
  *colorfulness = 0;
  for (int bin = 0; bin < kHueBins; ++bin) {
    float value = hue_histogram[bin] / hue_sum;
    if (value > 0.0f) {
      *colorfulness -= value * std::log(value);
    }
  }
  *colorfulness /= std::log(2.0f);

  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
