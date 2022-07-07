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
#include "mediapipe/calculators/landmarks/landmarks_to_mask_calculator.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

namespace mediapipe
{
  namespace
  {
    constexpr char kLandmarksTag[] = "LANDMARKS";
    constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
    constexpr char kVectorTag[] = "VECTOR";
    constexpr char kMaskTag[] = "MASK";
    constexpr char kFaceBoxTag[] = "FACEBOX";
    constexpr char kImageFrameTag[] = "IMAGE";

    std::unordered_map<std::string, const std::vector<int>> orderList = {
        {"UPPER_LIP", {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78}},
        {"LOWER_LIP", {61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146}},
        {"FACE_OVAL", {10, 338, 338, 297, 297, 332, 332, 284, 284, 251, 251, 389, 389, 356, 356, 454, 454, 323, 323, 361, 361, 288, 288, 397, 397, 365, 365, 379, 379, 378, 378, 400, 400, 377, 377, 152, 152, 148, 148, 176, 176, 149, 149, 150, 150, 136, 136, 172, 172, 58, 58, 132, 132, 93, 93, 234, 234, 127, 127, 162, 162, 21, 21, 54, 54, 103, 103, 67, 67, 109, 109, 10}},
        {"MOUTH_INSIDE", {78, 191, 80, 81, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95}},
        {"LEFT_EYE", {130, 33, 246, 161, 160, 159, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7}},
        {"RIGHT_EYE", {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382}},
        {"LEFT_BROW", {70, 63, 105, 66, 107, 55, 65, 52, 53, 46}},
        {"RIGHT_BROW", {336, 296, 334, 293, 301, 300, 283, 282, 295, 285}},
        {"LIPS", {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146}},
        {"PART_FOREHEAD_B", {21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 301, 293, 334, 296, 336, 9, 107, 66, 105, 63, 71}},
    };

    template <class LandmarkType>
    bool IsLandmarkVisibleAndPresent(const LandmarkType &landmark,
                                     bool utilize_visibility,
                                     float visibility_threshold,
                                     bool utilize_presence,
                                     float presence_threshold)
    {
      if (utilize_visibility && landmark.has_visibility() &&
          landmark.visibility() < visibility_threshold)
      {
        return false;
      }
      if (utilize_presence && landmark.has_presence() &&
          landmark.presence() < presence_threshold)
      {
        return false;
      }
      return true;
    }

    bool NormalizedtoPixelCoordinates(double normalized_x, double normalized_y,
                                      int image_width, int image_height, int *x_px,
                                      int *y_px)
    {
      CHECK(x_px != nullptr);
      CHECK(y_px != nullptr);
      CHECK_GT(image_width, 0);
      CHECK_GT(image_height, 0);

      if (normalized_x < 0 || normalized_x > 1.0 || normalized_y < 0 ||
          normalized_y > 1.0)
      {
        VLOG(1) << "Normalized coordinates must be between 0.0 and 1.0";
      }

      *x_px = static_cast<int32>(round(normalized_x * image_width));
      *y_px = static_cast<int32>(round(normalized_y * image_height));

      return true;
    }

    std::tuple<double, double, double, double> face_box;

    std::unique_ptr<cv::Mat> image_mat;

    int image_width_;
    int image_height_;

    float scale_factor_ = 1.0;

    bool image_frame_available_ = false;

  } // namespace

  absl::Status LandmarksToMaskCalculator::GetContract(
      CalculatorContract *cc)
  {
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ||
              cc->Inputs().HasTag(kNormLandmarksTag))
        << "None of the input streams are provided.";
    RET_CHECK(!(cc->Inputs().HasTag(kLandmarksTag) &&
                cc->Inputs().HasTag(kNormLandmarksTag)))
        << "Can only one type of landmark can be taken. Either absolute or "
           "normalized landmarks.";

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    if (cc->Inputs().HasTag(kLandmarksTag))
    {
      cc->Inputs().Tag(kLandmarksTag).Set<LandmarkList>();
    }
    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      cc->Inputs().Tag(kNormLandmarksTag).Set<NormalizedLandmarkList>();
    }
    if (cc->Outputs().HasTag(kMaskTag))
    {
      cc->Outputs().Tag(kMaskTag).Set<std::unordered_map<std::string, cv::Mat>>();
    }
    if (cc->Outputs().HasTag(kFaceBoxTag))
    {
      cc->Outputs().Tag(kFaceBoxTag).Set<std::tuple<double, double, double, double>>();
    }

    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      image_frame_available_ = true;
    }

    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::Process(CalculatorContext *cc)
  {
    // Check that landmarks are not empty and skip rendering if so.
    // Don't emit an empty packet for this timestamp.
    if (cc->Inputs().HasTag(kLandmarksTag) &&
        cc->Inputs().Tag(kLandmarksTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kNormLandmarksTag) &&
        cc->Inputs().Tag(kNormLandmarksTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.

    ImageFormat::Format target_format;
    std::unordered_map<std::string, cv::Mat> all_masks;

    MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));

    cv::Mat mat_image_ = *image_mat.get();
    image_width_ = image_mat->cols;
    image_height_ = image_mat->rows;

    MP_RETURN_IF_ERROR(GetMasks(cc, all_masks));

    MP_RETURN_IF_ERROR(GetFaceBox(cc));

    MP_RETURN_IF_ERROR(RenderToCpu(cc, all_masks));

    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::RenderToCpu(CalculatorContext *cc,
                                                      std::unordered_map<std::string, cv::Mat> &all_masks)
  {

    auto output_frame = absl::make_unique<std::unordered_map<std::string, cv::Mat>>(all_masks, all_masks.get_allocator());

    if (cc->Outputs().HasTag(kMaskTag))
    {
      cc->Outputs()
          .Tag(kMaskTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    auto output_frame2 = absl::make_unique<std::tuple<double, double, double, double>>(face_box);

    if (cc->Outputs().HasTag(kFaceBoxTag))
    {
      cc->Outputs()
          .Tag(kFaceBoxTag)
          .Add(output_frame2.release(), cc->InputTimestamp());
    }

    all_masks.clear();
    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::CreateRenderTargetCpu(
      CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat,
      ImageFormat::Format *target_format)
  {
    if (image_frame_available_)
    {
      const auto &input_frame =
          cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();

      int target_mat_type;
      switch (input_frame.Format())
      {
      case ImageFormat::SRGBA:
        *target_format = ImageFormat::SRGBA;
        target_mat_type = CV_8UC4;
        break;
      case ImageFormat::SRGB:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      case ImageFormat::GRAY8:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      default:
        return absl::UnknownError("Unexpected image frame format.");
        break;
      }

      image_mat = absl::make_unique<cv::Mat>(
          input_frame.Height(), input_frame.Width(), target_mat_type);

      auto input_mat = formats::MatView(&input_frame);

      if (input_frame.Format() == ImageFormat::GRAY8)
      {
        cv::Mat rgb_mat;
        cv::cvtColor(input_mat, rgb_mat, CV_GRAY2RGB);
        rgb_mat.copyTo(*image_mat);
      }
      else
      {
        input_mat.copyTo(*image_mat);
      }
    }
    else
    {
      image_mat = absl::make_unique<cv::Mat>(
          150, 150, CV_8UC4,
          cv::Scalar(255, 255,
                     255));
      *target_format = ImageFormat::SRGBA;
    }

    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::GetMasks(CalculatorContext *cc,
                                                   std::unordered_map<std::string, cv::Mat> &all_masks)
  {
    if (cc->Inputs().HasTag(kLandmarksTag))
    {
       const LandmarkList &landmarks =
          cc->Inputs().Tag(kNormLandmarksTag).Get<LandmarkList>();

      cv::Mat mask;
      std::vector<cv::Point> point_array;
      for (const auto &[key, value] : orderList)
      {
        for (auto order : value)
        {
          const Landmark &landmark = landmarks.landmark(order);

          if (!IsLandmarkVisibleAndPresent<Landmark>(
                  landmark, false,
                  0.0, false,
                  0.0))
          {
            continue;
          }

          const auto &point = landmark;
          int x = -1;
          int y = -1;
          CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                             image_height_, &x, &y));
          point_array.push_back(cv::Point(x, y));
        }

        std::vector<std::vector<cv::Point>> point_vec;
        point_vec.push_back(point_array);
       
        mask = cv::Mat::zeros(image_mat->size(), CV_32FC1);
        cv::fillPoly(mask, point_vec, cv::Scalar::all(255), cv::LINE_AA);
        mask.convertTo(mask, CV_8U);
        all_masks.insert({key, mask});
        point_vec.clear();
        point_array.clear();
      }
    }

    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      const NormalizedLandmarkList &landmarks =
          cc->Inputs().Tag(kNormLandmarksTag).Get<NormalizedLandmarkList>();

      cv::Mat mask;
      std::vector<cv::Point> point_array;
      for (const auto &[key, value] : orderList)
      {
        for (auto order : value)
        {
          const NormalizedLandmark &landmark = landmarks.landmark(order);

          if (!IsLandmarkVisibleAndPresent<NormalizedLandmark>(
                  landmark, false,
                  0.0, false,
                  0.0))
          {
            continue;
          }

          const auto &point = landmark;
          int x = -1;
          int y = -1;
          CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                             image_height_, &x, &y));
          point_array.push_back(cv::Point(x, y));
        }

        std::vector<std::vector<cv::Point>> point_vec;
        point_vec.push_back(point_array);
        mask = cv::Mat::zeros(image_mat->size(), CV_32FC1);
        cv::fillPoly(mask, point_vec, cv::Scalar::all(255), cv::LINE_AA);
        mask.convertTo(mask, CV_8U);
        all_masks.insert(make_pair(key, mask));
        point_vec.clear();
        point_array.clear();
      }
    }
    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::GetFaceBox(CalculatorContext *cc)
  {
    std::vector<int> x_s, y_s;
    double box_min_y, box_max_y, box_max_x, box_min_x;
    if (cc->Inputs().HasTag(kLandmarksTag))
    {
      const LandmarkList &landmarks =
          cc->Inputs().Tag(kLandmarksTag).Get<LandmarkList>();

      for (int i = 0; i < landmarks.landmark_size(); ++i)
      {
        const Landmark &landmark = landmarks.landmark(i);

        if (!IsLandmarkVisibleAndPresent<Landmark>(
                landmark, false,
                0.0, false,
                0.0))
        {
          continue;
        }

        const auto &point = landmark;
        int x = -1;
        int y = -1;
        CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                           image_height_, &x, &y));
        x_s.push_back(point.x());
        x_s.push_back(point.y());
      }
      cv::minMaxLoc(y_s, &box_min_y, &box_max_y);
      cv::minMaxLoc(x_s, &box_min_x, &box_max_x);
      box_min_y = box_min_y * 0.9;
      face_box = std::make_tuple(box_min_x, box_min_y, box_max_x, box_max_y);
    }

    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      const NormalizedLandmarkList &landmarks =
          cc->Inputs().Tag(kNormLandmarksTag).Get<NormalizedLandmarkList>();

      for (int i = 0; i < landmarks.landmark_size(); ++i)
      {
        const NormalizedLandmark &landmark = landmarks.landmark(i);

        if (!IsLandmarkVisibleAndPresent<NormalizedLandmark>(
                landmark, false,
                0.0, false,
                0.0))
        {
          continue;
        }

        const auto &point = landmark;
        int x = -1;
        int y = -1;
        CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                           image_height_, &x, &y));
        x_s.push_back(point.x());
        x_s.push_back(point.y());
      }
      cv::minMaxLoc(y_s, &box_min_y, &box_max_y);
      cv::minMaxLoc(x_s, &box_min_x, &box_max_x);
      box_min_y = box_min_y * 0.9;
      face_box = std::make_tuple(box_min_x, box_min_y, box_max_x, box_max_y);
    }

    return absl::OkStatus();
  }

  REGISTER_CALCULATOR(LandmarksToMaskCalculator);
} // namespace mediapipe
