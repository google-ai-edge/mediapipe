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
#include <iostream>

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
    constexpr char kFaceBoxTag[] = "FACEBOX";
    constexpr char kImageFrameTag[] = "IMAGE";
    constexpr char kImageNewTag[] = "IMAGE2";

    enum
    {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
      NUM_ATTRIBUTES
    };

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
  } // namespace

  class BilateralCalculator : public CalculatorBase
  {
  public:
    BilateralCalculator() = default;
    ~BilateralCalculator() override = default;

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
        uchar *data_image);

    absl::Status BilateralFilter(CalculatorContext *cc,
                                 const std::vector<double> &face_box);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    int image_width_;
    int image_height_;
    cv::Mat mat_image_;
    std::unique_ptr<cv::Mat> image_mat;
  };
  REGISTER_CALCULATOR(BilateralCalculator);

  absl::Status BilateralCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      CHECK(cc->Outputs().HasTag(kImageNewTag));
    }

    // Data streams to render.
    for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
         ++id)
    {
      auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
      std::string tag = tag_and_index.first;
      if (tag == kFaceBoxTag)
      {
        cc->Inputs().Get(id).Set<std::vector<double>>();
      }
      else if (tag.empty())
      {
        // Empty tag defaults to accepting a single object of Mat type.
        cc->Inputs().Get(id).Set<cv::Mat>();
      }
    }

    if (cc->Outputs().HasTag(kImageNewTag))
    {
      cc->Outputs().Tag(kImageNewTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    if (cc->Inputs().HasTag(kImageFrameTag) || HasImageTag(cc))
    {
      image_frame_available_ = true;
    }

    // Set the output header based on the input header (if present).
    const char *tag = kImageFrameTag;
    const char *out_tag = kImageNewTag;
    if (image_frame_available_ && !cc->Inputs().Tag(tag).Header().IsEmpty())
    {
      const auto &input_header =
          cc->Inputs().Tag(tag).Header().Get<VideoHeader>();
      auto *output_video_header = new VideoHeader(input_header);
      cc->Outputs().Tag(out_tag).SetHeader(Adopt(output_video_header));
    }

    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kFaceBoxTag) &&
        cc->Inputs().Tag(kFaceBoxTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;

    if (cc->Outputs().HasTag(kImageNewTag))
    {
      MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));
    }

    mat_image_ = *image_mat.get();
    image_width_ = image_mat->cols;
    image_height_ = image_mat->rows;

    const std::vector<double> &face_box =
        cc->Inputs().Tag(kFaceBoxTag).Get<std::vector<double>>();

    MP_RETURN_IF_ERROR(BilateralFilter(cc, face_box));

    // Copy the rendered image to output.
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr));

    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::RenderToCpu(
      CalculatorContext *cc, const ImageFormat::Format &target_format,
      uchar *data_image)
  {
    auto output_frame = absl::make_unique<ImageFrame>(
        target_format, image_width_, image_height_);

    output_frame->CopyPixelData(target_format, image_width_, image_height_, data_image,
                                ImageFrame::kDefaultAlignmentBoundary);

    if (cc->Outputs().HasTag(kImageNewTag))
    {
      cc->Outputs()
          .Tag(kImageNewTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::CreateRenderTargetCpu(
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
          cv::Scalar::all(255));
      *target_format = ImageFormat::SRGBA;
    }

    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::BilateralFilter(CalculatorContext *cc,
                                                const std::vector<double> &face_box)
  {
    cv::Mat patch_face = mat_image_(cv::Range(face_box[1], face_box[3]), 
                                    cv::Range(face_box[0], face_box[2]));
    cv::Mat patch_new, patch_wow;
    cv::cvtColor(patch_face, patch_wow, cv::COLOR_RGBA2RGB);
    cv::bilateralFilter(patch_wow, patch_new, 12, 50, 50);
    cv::cvtColor(patch_new, patch_new, cv::COLOR_RGB2RGBA);

    return absl::OkStatus();
  }

} // namespace mediapipe
