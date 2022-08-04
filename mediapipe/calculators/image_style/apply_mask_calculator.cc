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
#include <string>
#include <memory>

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
    constexpr char kImageFrameTag[] = "IMAGE";
    constexpr char kFakeBgTag[] = "FAKE_BG";
    constexpr char kLmMaskTag[] = "LM_MASK";

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }

    absl::StatusOr<cv::Mat> blend_mask(cv::Mat mask_face, cv::Mat mask_bbox, int kernel_size = 33, int reduce_size = 128)
    {
      int k_sz = kernel_size;
      auto [width, height] = mask_face.size();

      cv::Mat mask_face_0 = mask_face.clone();

      double K = (double)reduce_size / std::min(height, width);

      cv::resize(mask_face, mask_face, {(int)(width * K), (int)(height * K)});
      mask_face.convertTo(mask_face, CV_32F);

      cv::GaussianBlur(mask_face, mask_face, {k_sz, k_sz}, 0);
      mask_face *= 2;
      cv::threshold(mask_face, mask_face, 1, 255, CV_THRESH_TRUNC);

      cv::resize(mask_bbox, mask_bbox, {(int)(width * K), (int)(height * K)});

      mask_bbox.convertTo(mask_bbox, CV_32F);
      cv::GaussianBlur(mask_bbox, mask_bbox, {k_sz, k_sz}, 0);

      cv::Mat mask = mask_bbox.mul(mask_face);

      cv::Mat img_out;
      cv::resize(mask, img_out, {width, height});

      for (int i = 0; i < mask_face_0.rows; i++)
      {
        const uchar *ptr_mask_face = mask_face_0.ptr<uchar>(i);
        float *ptr_img_out = img_out.ptr<float>(i);
        for (int j = 0; j < mask_face_0.cols; j++)
        {
          if (ptr_mask_face[j] > 0)
          {
            ptr_img_out[j] = 1;
          }
        }
      }

      return img_out;
    }
  } // namespace

  class ApplyMaskCalculator : public CalculatorBase
  {
  public:
    ApplyMaskCalculator() = default;
    ~ApplyMaskCalculator() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status CreateRenderTargetCpu(CalculatorContext *cc,
                                       std::unique_ptr<cv::Mat> &image_mat,
                                       std::string_view tag,
                                       ImageFormat::Format *target_format);

    absl::Status RenderToCpu(
        CalculatorContext *cc, const ImageFormat::Format &target_format,
        uchar *data_image, std::unique_ptr<cv::Mat> &image_mat);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    int image_width_;
    int image_height_;
  };
  REGISTER_CALCULATOR(ApplyMaskCalculator);

  absl::Status ApplyMaskCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      CHECK(cc->Outputs().HasTag(kImageFrameTag));
    }

    if (cc->Inputs().HasTag(kFakeBgTag))
    {
      cc->Inputs().Tag(kFakeBgTag).Set<ImageFrame>();
    }
    if (cc->Inputs().HasTag(kLmMaskTag))
    {
      cc->Inputs().Tag(kLmMaskTag).Set<cv::Mat>();
    }
    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status ApplyMaskCalculator::Open(CalculatorContext *cc)
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

  absl::Status ApplyMaskCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;
    std::unique_ptr<cv::Mat> image_mat;
    MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, kImageFrameTag, &target_format));

    if (((cc->Inputs().HasTag(kFakeBgTag) &&
          !cc->Inputs().Tag(kFakeBgTag).IsEmpty())) &&
        ((cc->Inputs().HasTag(kLmMaskTag) &&
          !cc->Inputs().Tag(kLmMaskTag).IsEmpty())))
    {
      // Initialize render target, drawn with OpenCV.
      const auto &input_fake_bg = cc->Inputs().Tag(kFakeBgTag).Get<ImageFrame>();
      auto mat_fake_bg_ = formats::MatView(&input_fake_bg);

      cv::Mat lm_mask = cc->Inputs().Tag(kLmMaskTag).Get<cv::Mat>();
      
      cv::Mat mat_image_ = *image_mat.get();
      image_width_ = image_mat->cols;
      image_height_ = image_mat->rows;

      cv::Mat roi_mask = mat_image_.clone();

      cv::transform(roi_mask, roi_mask, cv::Matx13f(1, 1, 1));
      cv::threshold(roi_mask, roi_mask, 1, 255, CV_THRESH_TRUNC);

      ASSIGN_OR_RETURN(auto mask, blend_mask(lm_mask, roi_mask, 33));

      mat_image_.convertTo(mat_image_, CV_32F);
      mat_fake_bg_.convertTo(mat_fake_bg_, CV_32F);
      cv::resize(mat_fake_bg_, mat_fake_bg_, {image_width_, image_height_});
      cv::merge(std::vector{mask, mask, mask}, mask);

      cv::Mat im_out = mat_fake_bg_.mul(cv::Scalar::all(1) - mask) + mat_image_.mul(mask);

      im_out.convertTo(*image_mat, CV_8U);
    }
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr, image_mat));

    return absl::OkStatus();
  }

  absl::Status ApplyMaskCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status ApplyMaskCalculator::RenderToCpu(
      CalculatorContext *cc, const ImageFormat::Format &target_format,
      uchar *data_image, std::unique_ptr<cv::Mat> &image_mat)
  {
    auto output_frame = absl::make_unique<ImageFrame>(
        target_format, image_mat->cols, image_mat->rows);

    output_frame->CopyPixelData(target_format, image_mat->cols, image_mat->rows, data_image,
                                ImageFrame::kDefaultAlignmentBoundary);

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs()
          .Tag(kImageFrameTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status ApplyMaskCalculator::CreateRenderTargetCpu(
      CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat, std::string_view tag,
      ImageFormat::Format *target_format)
  {
    if (image_frame_available_)
    {
      const auto &input_frame =
          cc->Inputs().Tag(tag).Get<ImageFrame>();

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
          1920, 1080, CV_8UC4,
          cv::Scalar::all(255));
      *target_format = ImageFormat::SRGBA;
    }

    return absl::OkStatus();
  }
} // namespace mediapipe
