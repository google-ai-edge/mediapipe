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

#include <math.h>

#include <algorithm>
#include <cmath>
//#include <android/log.h>

#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"

namespace mediapipe
{
  namespace
  {

    constexpr char kMaskTag[] = "MASK";
    constexpr char kImageFrameTag[] = "IMAGE";

    enum
    {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
      NUM_ATTRIBUTES
    };

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
    absl::Status CreateRenderTargetCpu(CalculatorContext *cc,
                                       std::unique_ptr<cv::Mat> &image_mat,
                                       ImageFormat::Format *target_format);

    absl::Status RenderToCpu(
        CalculatorContext *cc, const ImageFormat::Format &target_format,
        uchar *data_image, std::unique_ptr<cv::Mat> &image_mat);

    absl::Status WhitenTeeth(CalculatorContext *cc, ImageFormat::Format *target_format,
                             const std::unordered_map<std::string, cv::Mat> &mask_vec);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    std::unique_ptr<cv::Mat> image_mat;
    cv::Mat mat_image_;
    int image_width_;
    int image_height_;
  };
  REGISTER_CALCULATOR(WhitenTeethCalculator);

  absl::Status WhitenTeethCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      CHECK(cc->Outputs().HasTag(kImageFrameTag));
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

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    if (cc->Inputs().HasTag(kImageFrameTag) || HasImageTag(cc))
    {
      image_frame_available_ = true;
    }
    else
    {
    }

    // Set the output header based on the input header (if present).
    const char *tag = kImageFrameTag;
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
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kMaskTag) &&
        cc->Inputs().Tag(kMaskTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));
    }
    mat_image_ = *image_mat.get();
    image_width_ = image_mat->cols;
    image_height_ = image_mat->rows;
    
    const std::vector<std::unordered_map<std::string, cv::Mat>> &mask_vec =
        cc->Inputs().Tag(kMaskTag).Get<std::vector<std::unordered_map<std::string, cv::Mat>>>();
    if (mask_vec.size() > 0)
    {
      for (auto mask : mask_vec)
        MP_RETURN_IF_ERROR(WhitenTeeth(cc, &target_format, mask));
    }

    // Copy the rendered image to output.
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr, image_mat));

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::RenderToCpu(
      CalculatorContext *cc, const ImageFormat::Format &target_format,
      uchar *data_image, std::unique_ptr<cv::Mat> &image_mat)
  {

    cv::Mat mat_image_ = *image_mat.get();

    auto output_frame = absl::make_unique<ImageFrame>(
        target_format, image_width_, image_height_);

    output_frame->CopyPixelData(target_format, image_width_, image_height_, data_image,
                                ImageFrame::kDefaultAlignmentBoundary);

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs()
          .Tag(kImageFrameTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status WhitenTeethCalculator::CreateRenderTargetCpu(
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

  absl::Status WhitenTeethCalculator::WhitenTeeth(CalculatorContext *cc,
                                                  ImageFormat::Format *target_format,
                                                  const std::unordered_map<std::string, cv::Mat> &mask_vec)
  {
    cv::Mat mouth_mask, mouth;
    mouth_mask = cv::Mat::zeros(mat_image_.size(), CV_32F);

    mouth_mask = mask_vec.find("MOUTH_INSIDE")->second.clone();

    cv::resize(mouth_mask, mouth, mat_image_.size(), cv::INTER_LINEAR);

    std::vector<int> x, y;
    std::vector<cv::Point> location;

    cv::findNonZero(mouth, location);

    for (auto &i : location)
    {
      x.push_back(i.x);
      y.push_back(i.y);
    }

    if (!(x.empty()) && !(y.empty()))
    {
      double mouth_min_y, mouth_max_y, mouth_max_x, mouth_min_x;
      cv::minMaxLoc(y, &mouth_min_y, &mouth_max_y);
      cv::minMaxLoc(x, &mouth_min_x, &mouth_max_x);
      double mh = mouth_max_y - mouth_min_y;
      double mw = mouth_max_x - mouth_min_x;
      cv::Mat mouth_crop_mask;
      mouth.convertTo(mouth, CV_32F, 1.0 / 255);
      mouth.convertTo(mouth, CV_32F, 1.0 / 255);
      if (mh / mw > 0.17)
      {
        mouth_min_y = static_cast<int>(std::max(mouth_min_y - mh * 0.1, 0.0));
        mouth_max_y = static_cast<int>(std::min(mouth_max_y + mh * 0.1, (double)image_height_));
        mouth_min_x = static_cast<int>(std::max(mouth_min_x - mw * 0.1, 0.0));
        mouth_max_x = static_cast<int>(std::min(mouth_max_x + mw * 0.1, (double)image_width_));
        mouth_crop_mask = mouth(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x));
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
