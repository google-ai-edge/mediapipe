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
//
// This Calculator takes an ImageFrame and scales it appropriately.

#include <algorithm>
#include <memory>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/border_detection_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

using mediapipe::Adopt;
using mediapipe::CalculatorBase;
using mediapipe::ImageFrame;
using mediapipe::PacketTypeSet;
using mediapipe::autoflip::Border;

constexpr char kDetectedBorders[] = "DETECTED_BORDERS";
constexpr int kMinBorderDistance = 5;
constexpr int kKMeansClusterCount = 4;
constexpr int kMaxPixelsToProcess = 300000;
constexpr char kVideoInputTag[] = "VIDEO";

namespace mediapipe {
namespace autoflip {

namespace {

// Sets rect values into a proto.
void SetRect(const cv::Rect& region,
             const Border::RelativePosition& relative_position, Border* part) {
  part->mutable_border_position()->set_x(region.x);
  part->mutable_border_position()->set_y(region.y);
  part->mutable_border_position()->set_width(region.width);
  part->mutable_border_position()->set_height(region.height);
  part->set_relative_position(relative_position);
}

}  // namespace

// This calculator takes a sequence of images (video) and detects solid color
// borders as well as the dominant color of the non-border area.  This per-frame
// information is passed to downstream calculators.
class BorderDetectionCalculator : public CalculatorBase {
 public:
  BorderDetectionCalculator() : frame_width_(-1), frame_height_(-1) {}
  ~BorderDetectionCalculator() override {}
  BorderDetectionCalculator(const BorderDetectionCalculator&) = delete;
  BorderDetectionCalculator& operator=(const BorderDetectionCalculator&) =
      delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  // Given a color and image direction, check to see if a border of that color
  // exists.
  void DetectBorder(const cv::Mat& frame, const Color& color,
                    const Border::RelativePosition& direction,
                    StaticFeatures* features);

  // Provide the percent this color shows up in a given image.
  double ColorCount(const Color& mask_color, const cv::Mat& image) const;

  // Set member vars (image size) and confirm no changes frame-to-frame.
  absl::Status SetAndCheckInputs(const cv::Mat& frame);

  // Find the dominant color for a input image.
  double FindDominantColor(const cv::Mat& image, Color* dominant_color);

  // Frame width and height.
  int frame_width_;
  int frame_height_;

  // Options for processing.
  BorderDetectionCalculatorOptions options_;
};
REGISTER_CALCULATOR(BorderDetectionCalculator);

absl::Status BorderDetectionCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<BorderDetectionCalculatorOptions>();
  RET_CHECK_LT(options_.vertical_search_distance(), 0.5)
      << "Search distance must be less than half the full image.";
  return absl::OkStatus();
}

absl::Status BorderDetectionCalculator::SetAndCheckInputs(
    const cv::Mat& frame) {
  if (frame_width_ < 0) {
    frame_width_ = frame.cols;
  }
  if (frame_height_ < 0) {
    frame_height_ = frame.rows;
  }
  RET_CHECK_EQ(frame.cols, frame_width_)
      << "Input frame dimensions must remain constant throughout the video.";
  RET_CHECK_EQ(frame.rows, frame_height_)
      << "Input frame dimensions must remain constant throughout the video.";
  RET_CHECK_EQ(frame.channels(), 3) << "Input video type must be 3-channel";
  return absl::OkStatus();
}

absl::Status BorderDetectionCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  if (!cc->Inputs().HasTag(kVideoInputTag) ||
      cc->Inputs().Tag(kVideoInputTag).Value().IsEmpty()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Input tag VIDEO not set or empty at timestamp: "
           << cc->InputTimestamp().Value();
  }
  cv::Mat frame = mediapipe::formats::MatView(
      &cc->Inputs().Tag(kVideoInputTag).Get<ImageFrame>());
  MP_RETURN_IF_ERROR(SetAndCheckInputs(frame));

  // Initialize output and set default values.
  std::unique_ptr<StaticFeatures> features =
      absl::make_unique<StaticFeatures>();
  features->mutable_non_static_area()->set_x(0);
  features->mutable_non_static_area()->set_width(frame_width_);
  features->mutable_non_static_area()->set_y(options_.default_padding_px());
  features->mutable_non_static_area()->set_height(
      std::max(0, frame_height_ - options_.default_padding_px() * 2));

  // Check for border at the top of the frame.
  Color seed_color_top;
  FindDominantColor(frame(cv::Rect(0, 0, frame_width_, 1)), &seed_color_top);
  DetectBorder(frame, seed_color_top, Border::TOP, features.get());

  // Check for border at the bottom of the frame.
  Color seed_color_bottom;
  FindDominantColor(frame(cv::Rect(0, frame_height_ - 1, frame_width_, 1)),
                    &seed_color_bottom);
  DetectBorder(frame, seed_color_bottom, Border::BOTTOM, features.get());

  // Check the non-border area for a dominant color.
  cv::Mat non_static_frame = frame(
      cv::Rect(features->non_static_area().x(), features->non_static_area().y(),
               features->non_static_area().width(),
               features->non_static_area().height()));
  Color dominant_color_nonborder;
  double dominant_color_percent =
      FindDominantColor(non_static_frame, &dominant_color_nonborder);
  if (dominant_color_percent > options_.solid_background_tol_perc()) {
    auto* bg_color = features->mutable_solid_background();
    bg_color->set_r(dominant_color_nonborder.r());
    bg_color->set_g(dominant_color_nonborder.g());
    bg_color->set_b(dominant_color_nonborder.b());
  }

  // Output result.
  cc->Outputs()
      .Tag(kDetectedBorders)
      .AddPacket(Adopt(features.release()).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

//  Find the dominant color within an image.
double BorderDetectionCalculator::FindDominantColor(const cv::Mat& image_raw,
                                                    Color* dominant_color) {
  cv::Mat image;
  if (image_raw.total() > kMaxPixelsToProcess) {
    float resize = kMaxPixelsToProcess / static_cast<float>(image_raw.total());
    cv::resize(image_raw, image, cv::Size(), resize, resize);
  } else {
    image = image_raw;
  }

  cv::Mat float_data, cluster, cluster_center;
  image.convertTo(float_data, CV_32F);
  cv::Mat reshaped = float_data.reshape(1, float_data.total());

  cv::kmeans(reshaped, kKMeansClusterCount, cluster,
             cv::TermCriteria(CV_TERMCRIT_ITER, 5, 1.0), 1,
             cv::KMEANS_PP_CENTERS, cluster_center);

  std::vector<int> count(kKMeansClusterCount, 0);
  for (int i = 0; i < cluster.rows; i++) {
    count[cluster.at<int>(i, 0)]++;
  }
  auto max_cluster_ptr = std::max_element(count.begin(), count.end());
  double max_cluster_perc =
      *max_cluster_ptr / static_cast<double>(cluster.rows);
  int max_cluster_idx = std::distance(count.begin(), max_cluster_ptr);

  dominant_color->set_r(cluster_center.at<float>(max_cluster_idx, 2));
  dominant_color->set_g(cluster_center.at<float>(max_cluster_idx, 1));
  dominant_color->set_b(cluster_center.at<float>(max_cluster_idx, 0));

  return max_cluster_perc;
}

double BorderDetectionCalculator::ColorCount(const Color& mask_color,
                                             const cv::Mat& image) const {
  int background_count = 0;
  for (int i = 0; i < image.rows; i++) {
    const uint8* row_ptr = image.ptr<uint8>(i);
    for (int j = 0; j < image.cols * 3; j += 3) {
      if (std::abs(mask_color.r() - static_cast<int>(row_ptr[j + 2])) <=
              options_.color_tolerance() &&
          std::abs(mask_color.g() - static_cast<int>(row_ptr[j + 1])) <=
              options_.color_tolerance() &&
          std::abs(mask_color.b() - static_cast<int>(row_ptr[j])) <=
              options_.color_tolerance()) {
        background_count++;
      }
    }
  }
  return background_count / static_cast<double>(image.rows * image.cols);
}

void BorderDetectionCalculator::DetectBorder(
    const cv::Mat& frame, const Color& color,
    const Border::RelativePosition& direction, StaticFeatures* features) {
  // Search the entire image until we find an object, or hit the max search
  // distance.
  int search_distance =
      (direction == Border::TOP || direction == Border::BOTTOM) ? frame.rows
                                                                : frame.cols;
  search_distance *= options_.vertical_search_distance();

  // Check if each next line has a dominant color that matches the given
  // border color.
  int last_border = -1;
  for (int i = 0; i < search_distance; i++) {
    cv::Rect current_row;
    switch (direction) {
      case Border::TOP:
        current_row = cv::Rect(0, i, frame.cols, 1);
        break;
      case Border::BOTTOM:
        current_row = cv::Rect(0, frame.rows - i - 1, frame.cols, 1);
        break;
    }
    if (ColorCount(color, frame(current_row)) <
        options_.border_color_pixel_perc()) {
      break;
    }
    last_border = i;
  }

  // Reject results that are not borders (or too small).
  if (last_border <= kMinBorderDistance || last_border == search_distance - 1) {
    return;
  }

  // Apply defined padding.
  last_border += options_.border_object_padding_px();

  switch (direction) {
    case Border::TOP:
      SetRect(cv::Rect(0, 0, frame.cols, last_border), Border::TOP,
              features->add_border());
      features->mutable_non_static_area()->set_y(
          last_border + features->non_static_area().y());
      features->mutable_non_static_area()->set_height(
          std::max(0, frame_height_ - (features->non_static_area().y() +
                                       options_.default_padding_px())));
      break;
    case Border::BOTTOM:
      SetRect(
          cv::Rect(0, frame.rows - last_border - 1, frame.cols, last_border),
          Border::BOTTOM, features->add_border());

      features->mutable_non_static_area()->set_height(std::max(
          0, frame.rows - (features->non_static_area().y() + last_border +
                           options_.default_padding_px())));

      break;
  }
}

absl::Status BorderDetectionCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kVideoInputTag).Set<ImageFrame>();
  cc->Outputs().Tag(kDetectedBorders).Set<StaticFeatures>();
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
