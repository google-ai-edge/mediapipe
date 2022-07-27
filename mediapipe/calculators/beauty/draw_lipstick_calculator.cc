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
#include <map>
#include <string>

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

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
  } // namespace

  class DrawLipstickCalculator : public CalculatorBase
  {
  public:
    DrawLipstickCalculator() = default;
    ~DrawLipstickCalculator() override = default;

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

    absl::Status DrawLipstick(CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat,
                              ImageFormat::Format *target_format,
                              const std::unordered_map<std::string, cv::Mat> &mask_vec);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    std::unordered_map<std::string, cv::Mat> all_masks;
  };
  REGISTER_CALCULATOR(DrawLipstickCalculator);

  absl::Status DrawLipstickCalculator::GetContract(CalculatorContract *cc)
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

  absl::Status DrawLipstickCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    if (cc->Inputs().HasTag(kImageFrameTag) || HasImageTag(cc))
    {
      image_frame_available_ = true;
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

  absl::Status DrawLipstickCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.
    std::unique_ptr<cv::Mat> image_mat;
    ImageFormat::Format target_format;

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));
    }

    if (cc->Inputs().HasTag(kMaskTag) &&
        !cc->Inputs().Tag(kMaskTag).IsEmpty())
    {
      const std::vector<std::unordered_map<std::string, cv::Mat>> &mask_vec =
          cc->Inputs().Tag(kMaskTag).Get<std::vector<std::unordered_map<std::string, cv::Mat>>>();
      if (mask_vec.size() > 0)
      {
        for (auto mask : mask_vec)
          MP_RETURN_IF_ERROR(DrawLipstick(cc, image_mat, &target_format, mask));
      }
    }
    // Copy the rendered image to output.
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr, image_mat));

    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::RenderToCpu(
      CalculatorContext *cc, const ImageFormat::Format &target_format,
      uchar *data_image, std::unique_ptr<cv::Mat> &image_mat)
  {
    cv::Mat mat_image_ = *image_mat.get();

    auto output_frame = absl::make_unique<ImageFrame>(
        target_format, mat_image_.cols, mat_image_.rows);

    output_frame->CopyPixelData(target_format, mat_image_.cols, mat_image_.rows, data_image,
                                ImageFrame::kDefaultAlignmentBoundary);

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs()
          .Tag(kImageFrameTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::CreateRenderTargetCpu(
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
          cv::Scalar(cv::Scalar::all(255)));
      *target_format = ImageFormat::SRGBA;
    }

    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::DrawLipstick(CalculatorContext *cc,
                                                    std::unique_ptr<cv::Mat> &image_mat,
                                                    ImageFormat::Format *target_format,
                                                    const std::unordered_map<std::string, cv::Mat> &mask_vec)
  {
    cv::Mat mat_image__ = *image_mat.get();

    cv::Mat spec_lips_mask, upper_lips_mask, lower_lips_mask;
    spec_lips_mask = cv::Mat::zeros(mat_image__.size(), CV_32F);
    upper_lips_mask = cv::Mat::zeros(mat_image__.size(), CV_32F);
    lower_lips_mask = cv::Mat::zeros(mat_image__.size(), CV_32F);

    upper_lips_mask = mask_vec.find("UPPER_LIP")->second;
    lower_lips_mask = mask_vec.find("LOWER_LIP")->second;

    spec_lips_mask = upper_lips_mask + lower_lips_mask;

    spec_lips_mask.convertTo(spec_lips_mask, CV_8U);

    cv::resize(spec_lips_mask, spec_lips_mask, mat_image__.size(), cv::INTER_LINEAR);

    std::vector<int> x, y;
    std::vector<cv::Point> location;

    cv::findNonZero(spec_lips_mask, location);

    for (auto &i : location)
    {
      x.push_back(i.x);
      y.push_back(i.y);
    }

    if (!(x.empty()) && !(y.empty()))
    {

      double min_y, max_y, max_x, min_x;
      cv::minMaxLoc(y, &min_y, &max_y);
      cv::minMaxLoc(x, &min_x, &max_x);

      cv::Mat lips_crop_mask = spec_lips_mask(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
      lips_crop_mask.convertTo(lips_crop_mask, CV_32F, 1.0 / 255);

      cv::Mat lips_crop = cv::Mat(mat_image__(cv::Range(min_y, max_y), cv::Range(min_x, max_x)));

      cv::Mat lips_blend = cv::Mat(lips_crop.size().height, lips_crop.size().width, CV_32FC4, cv::Scalar(255.0, 0, 0, 0));

      std::vector<cv::Mat> channels(4);

      cv::split(lips_blend, channels);
      channels[3] = lips_crop_mask * 20;

      cv::merge(channels, lips_blend);

      cv::Mat tmp_lip_mask;

      channels[3].convertTo(tmp_lip_mask, CV_32FC1, 1.0 / 255);

      cv::split(lips_blend, channels);
      for (auto &ch : channels)
      {
        cv::multiply(ch, tmp_lip_mask, ch, 1.0, CV_32F);
      }
      cv::merge(channels, lips_blend);

      cv::subtract(1.0, tmp_lip_mask, tmp_lip_mask, cv::noArray(), CV_32F);

      cv::split(lips_crop, channels);
      for (auto &ch : channels)
      {
        cv::multiply(ch, tmp_lip_mask, ch, 1.0, CV_8U);
      }
      cv::merge(channels, lips_crop);

      cv::add(lips_blend, lips_crop, lips_crop, cv::noArray(), CV_8U);

      lips_crop = cv::abs(lips_crop);

      cvtColor(lips_crop, lips_crop, cv::COLOR_RGBA2RGB);

      cv::Mat slice = mat_image__(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
      lips_crop_mask.convertTo(lips_crop_mask, slice.type());
      slice.copyTo(slice, lips_crop_mask);

      cv::Mat masked_lips_crop, slice_gray;
      lips_crop.copyTo(masked_lips_crop, lips_crop_mask);

      cv::cvtColor(masked_lips_crop, slice_gray, cv::COLOR_RGB2GRAY);

      masked_lips_crop.copyTo(slice, slice_gray);
    }

    return absl::OkStatus();
  }

} // namespace mediapipe
