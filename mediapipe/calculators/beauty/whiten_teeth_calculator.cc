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

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"

using namespace std;
namespace mediapipe
{
  namespace
  {

    constexpr char kMaskTag[] = "MASK";
    constexpr char kMatTag[] = "MAT";

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
  } // namespace

  class WhitenTeethCalculator : public CalculatorBase
  {
  public:
    WhitenTeethCalculator() = default;
    ~WhitenTeethCalculator() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status RenderToCpu(
        CalculatorContext *cc);

    absl::Status WhitenTeeth(CalculatorContext *cc,
                             const std::unordered_map<std::string, cv::Mat> &mask_vec);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    cv::Mat mouth;
    cv::Mat mat_image_;
    int image_width_;
    int image_height_;
  };
  REGISTER_CALCULATOR(WhitenTeethCalculator);

  absl::Status WhitenTeethCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kMatTag))
    {
      cc->Inputs().Tag(kMatTag).Set<cv::Mat>();
      CHECK(cc->Outputs().HasTag(kMatTag));
    }

    // Data streams to render.
    for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
         ++id)
    {
      auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
      std::string tag = tag_and_index.first;
      if (tag == kMaskTag)
      {
        cc->Inputs().Get(id).Set<std::vector<std::unordered_map<std::string, cv::Mat>>>();
      }
      else if (tag.empty())
      {
        // Empty tag defaults to accepting a single object of Mat type.
        cc->Inputs().Get(id).Set<cv::Mat>();
      }
    }

    if (cc->Outputs().HasTag(kMatTag))
    {
      cc->Outputs().Tag(kMatTag).Set<cv::Mat>();
    }
    if (cc->Outputs().HasTag(kMaskTag))
    {
      cc->Outputs().Tag(kMaskTag).Set<cv::Mat>();
    }

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    if (cc->Inputs().HasTag(kMatTag) || HasImageTag(cc))
    {
      image_frame_available_ = true;
    }

    // Set the output header based on the input header (if present).
    const char *tag = kMatTag;
    if (image_frame_available_ && !cc->Inputs().Tag(tag).Header().IsEmpty())
    {
      const auto &input_header =
          cc->Inputs().Tag(tag).Header().Get<VideoHeader>();
      auto *output_video_header = new VideoHeader(input_header);
      cc->Outputs().Tag(tag).SetHeader(Adopt(output_video_header));
    }

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kMatTag) &&
        cc->Inputs().Tag(kMatTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    ImageFormat::Format target_format;

    const cv::Mat &input_mat =
        cc->Inputs().Tag(kMatTag).Get<cv::Mat>();

    mat_image_ = input_mat.clone();

    image_width_ = input_mat.cols;
    image_height_ = input_mat.rows;

    if (cc->Inputs().HasTag(kMaskTag) &&
        !cc->Inputs().Tag(kMaskTag).IsEmpty())
    {
      const std::vector<std::unordered_map<std::string, cv::Mat>> &mask_vec =
          cc->Inputs().Tag(kMaskTag).Get<std::vector<std::unordered_map<std::string, cv::Mat>>>();
      if (mask_vec.size() > 0)
      {
        for (auto mask : mask_vec)
          MP_RETURN_IF_ERROR(WhitenTeeth(cc, mask));
      }
    }

    MP_RETURN_IF_ERROR(RenderToCpu(cc));

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::RenderToCpu(
      CalculatorContext *cc)
  {
    auto output_frame = absl::make_unique<cv::Mat>(mat_image_);

    if (cc->Outputs().HasTag(kMatTag))
    {
      cc->Outputs()
          .Tag(kMatTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    mouth.convertTo(mouth, CV_32F, 255);
    auto output_frame2 = absl::make_unique<cv::Mat>(mouth);

    if (cc->Outputs().HasTag(kMaskTag))
    {
      cc->Outputs()
          .Tag(kMaskTag)
          .Add(output_frame2.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::WhitenTeeth(CalculatorContext *cc,
                                                  const std::unordered_map<std::string, cv::Mat> &mask_vec)
  {
    cv::Mat mouth_mask = mask_vec.find("MOUTH_INSIDE")->second.clone();

    cv::resize(mouth_mask, mouth, mat_image_.size(), cv::INTER_LINEAR);

    cv::Rect rect = cv::boundingRect(mouth);

    if (!rect.empty())
    {
      double mouth_min_y = rect.y, mouth_max_y = rect.y + rect.height,
             mouth_max_x = rect.x + rect.width, mouth_min_x = rect.x;

      double mh = mouth_max_y - mouth_min_y;
      double mw = mouth_max_x - mouth_min_x;

      mouth.convertTo(mouth, CV_32F, 1.0 / 255);
      mouth.convertTo(mouth, CV_32F, 1.0 / 255);

      if (mh / mw > 0.17)
      {
        mouth_min_y = static_cast<int>(std::max(mouth_min_y - mh * 0.1, 0.0));
        mouth_max_y = static_cast<int>(std::min(mouth_max_y + mh * 0.1, (double)image_height_));
        mouth_min_x = static_cast<int>(std::max(mouth_min_x - mw * 0.1, 0.0));
        mouth_max_x = static_cast<int>(std::min(mouth_max_x + mw * 0.1, (double)image_width_));
        cv::Mat mouth_crop_mask = mouth(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x));
        cv::Mat img_hsv, tmp_mask, img_hls;
        cv::cvtColor(mat_image_(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x)), img_hsv,
                     cv::COLOR_RGBA2RGB);
        cv::cvtColor(img_hsv, img_hsv,
                     cv::COLOR_RGB2HSV);

        cv::Mat _mouth_erode_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(7, 7));

        cv::erode(mouth_crop_mask * 255, tmp_mask, _mouth_erode_kernel, cv::Point(-1, -1), 3);
        cv::GaussianBlur(tmp_mask, tmp_mask, cv::Size(51, 51), 0);

        img_hsv.convertTo(img_hsv, CV_8U);

        std::vector<cv::Mat> channels(3);
        cv::split(img_hsv, channels);

        cv::Mat tmp;
        cv::multiply(channels[1], tmp_mask, tmp, 0.3, CV_8U);
        cv::subtract(channels[1], tmp, channels[1], cv::noArray(), CV_8U);
        channels[1] = cv::min(255, channels[1]);
        cv::merge(channels, img_hsv);

        cv::cvtColor(img_hsv, img_hsv, cv::COLOR_HSV2RGB);
        cv::cvtColor(img_hsv, img_hls, cv::COLOR_RGB2HLS);

        cv::split(img_hls, channels);
        cv::multiply(channels[1], tmp_mask, tmp, 0.3, CV_8U);
        cv::add(channels[1], tmp, channels[1], cv::noArray(), CV_8U);
        channels[1] = cv::min(255, channels[1]);
        cv::merge(channels, img_hls);

        cv::cvtColor(img_hls, img_hls, cv::COLOR_HLS2RGB);
        cv::cvtColor(img_hls, img_hls, cv::COLOR_RGB2RGBA);

        cv::Mat slice = mat_image_(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x));
        img_hls.copyTo(slice);
      }
    }

    return absl::OkStatus();
  }

} // namespace mediapipe
