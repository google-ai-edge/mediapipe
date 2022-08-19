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

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

using namespace std;
namespace mediapipe
{
  namespace
  {
    constexpr char kLandmarksTag[] = "LANDMARKS";
    constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
    constexpr char kVectorTag[] = "VECTOR";
    constexpr char kMaskTag[] = "MASK";
    constexpr char kFaceBoxTag[] = "FACEBOX";
    constexpr char kImageSizeTag[] = "SIZE";

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

    int image_width_;
    int image_height_;

    float scale_factor_ = 1.0;
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

    if (cc->Inputs().HasTag(kImageSizeTag))
    {
      cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
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

    return absl::OkStatus();
  }

  absl::Status LandmarksToMaskCalculator::Process(CalculatorContext *cc)
  {
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

    std::unordered_map<std::string, cv::Mat> all_masks;

    const auto size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    image_width_ = size.first;
    image_height_ = size.second;
    CHECK_GT(image_width_, 0);
    CHECK_GT(image_height_, 0);

    MP_RETURN_IF_ERROR(GetMasks(cc, all_masks));

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

  absl::Status LandmarksToMaskCalculator::GetMasks(CalculatorContext *cc,
                                                   std::unordered_map<std::string, cv::Mat> &all_masks)
  {
    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      std::vector<int> x_s, y_s;
      double box_min_y, box_max_y, box_max_x, box_min_x;
      const NormalizedLandmarkList &landmarks =
          cc->Inputs().Tag(kNormLandmarksTag).Get<NormalizedLandmarkList>();

      cv::Mat mask;
      std::vector<cv::Point> point_array;
      std::vector<std::vector<cv::Point>> point_vec;
      for (const auto &[key, value] : orderList)
      {
        point_vec = {};
        point_array = {};
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
        cv::Mat po(point_array);
        po.convertTo(po, CV_32F);
        cv::Mat min, max;

        cv::reduce(po, min, 0, CV_REDUCE_MIN, CV_32F);
        cv::reduce(po, max, 0, CV_REDUCE_MAX, CV_32F);

        min.at<float>(0,1)*=0.9;
        face_box = {min.at<float>(0,0), min.at<float>(0,1), max.at<float>(0,0), max.at<float>(0,1)};

        point_vec.push_back(point_array);
        mask = cv::Mat::zeros({image_width_, image_height_}, CV_32FC1);
        cv::fillPoly(mask, point_vec, cv::Scalar::all(255), cv::LINE_AA);
        mask.convertTo(mask, CV_8U);
        all_masks.insert({key, mask});
      }
    }
    return absl::OkStatus();
  }

  REGISTER_CALCULATOR(LandmarksToMaskCalculator);
} // namespace mediapipe
