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

    enum
    {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
      NUM_ATTRIBUTES
    };

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
  } // namespace

  class SmoothFaceCalculator : public CalculatorBase
  {
  public:
    SmoothFaceCalculator() = default;
    ~SmoothFaceCalculator() override = default;

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

    absl::Status SmoothFace(CalculatorContext *cc,
                            ImageFormat::Format *target_format,
                            const std::unordered_map<std::string, cv::Mat> &mask_vec,
                            const std::tuple<double, double, double, double> &face_box);

    cv::Mat predict_forehead_mask(const std::unordered_map<std::string, cv::Mat> &mask_vec, double face_box_min_y);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    int image_width_;
    int image_height_;
    cv::Mat mat_image_;
    std::unique_ptr<cv::Mat> image_mat;
  };
  REGISTER_CALCULATOR(SmoothFaceCalculator);

  absl::Status SmoothFaceCalculator::GetContract(CalculatorContract *cc)
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

      if (tag == kFaceBoxTag)
      {
        cc->Inputs().Get(id).Set<std::vector<std::tuple<double, double, double, double>>>();
      }
    }

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator::Open(CalculatorContext *cc)
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

  absl::Status SmoothFaceCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
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

    if (cc->Inputs().HasTag(kMaskTag) &&
        !cc->Inputs().Tag(kMaskTag).IsEmpty() &&
        cc->Inputs().HasTag(kFaceBoxTag) &&
        !cc->Inputs().Tag(kFaceBoxTag).IsEmpty())
    {
      const std::vector<std::unordered_map<std::string, cv::Mat>> &mask_vec =
          cc->Inputs().Tag(kMaskTag).Get<std::vector<std::unordered_map<std::string, cv::Mat>>>();

      const std::vector<std::tuple<double, double, double, double>> &face_box =
          cc->Inputs().Tag(kFaceBoxTag).Get<std::vector<std::tuple<double, double, double, double>>>();

      if (mask_vec.size() > 0 && face_box.size() > 0)
      {
        for (int i = 0; i < mask_vec.size(); i++)
          MP_RETURN_IF_ERROR(SmoothFace(cc, &target_format, mask_vec[i], face_box[i]));
      }
    }
    // Copy the rendered image to output.
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr));

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator::RenderToCpu(
      CalculatorContext *cc, const ImageFormat::Format &target_format,
      uchar *data_image)
  {
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

  absl::Status SmoothFaceCalculator::CreateRenderTargetCpu(
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

  cv::Mat SmoothFaceCalculator::predict_forehead_mask(const std::unordered_map<std::string, cv::Mat> &mask_vec, double face_box_min_y)
  {

    cv::Mat part_forehead_mask = mask_vec.find("PART_FOREHEAD_B")->second.clone();
    part_forehead_mask.convertTo(part_forehead_mask, CV_32F, 1.0 / 255);
    part_forehead_mask.convertTo(part_forehead_mask, CV_8U);

    cv::Mat image_sm, image_sm_hsv, skinMask;

    cv::resize(mat_image_, image_sm, cv::Size(image_width_, image_height_));
    cv::cvtColor(image_sm, image_sm_hsv, cv::COLOR_BGR2HSV);

    std::vector<int> x, y;
    std::vector<cv::Point> location;

    cv::Vec3d hsv_min, hsv_max;

    std::vector<cv::Mat> channels(3);
    cv::split(image_sm_hsv, channels);
    std::vector<std::vector<double>> minx(3), maxx(3);
    int c = 0;
    for (auto ch : channels)
    {
      cv::Mat row, mask_row;
      double min, max;
      for (int i = 0; i < ch.rows; i++)
      {
        row = ch.row(i);
        mask_row = part_forehead_mask.row(i);
        cv::minMaxLoc(row, &min, &max, 0, 0, mask_row);
        minx[c].push_back(min);
        maxx[c].push_back(max);
      }
      c++;
    }
    for (int i = 0; i < 3; i++)
    {
      hsv_min[i] = *std::min_element(minx[i].begin(), minx[i].end());
    }
    for (int i = 0; i < 3; i++)
    {
      hsv_max[i] = *std::max_element(maxx[i].begin(), maxx[i].end());
    }

    cv::Mat _forehead_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
    cv::inRange(image_sm_hsv, hsv_min, hsv_max, skinMask);
    cv::erode(skinMask, skinMask, _forehead_kernel, cv::Point(-1, -1), 2);
    cv::dilate(skinMask, skinMask, _forehead_kernel, cv::Point(-1, -1), 2);
    skinMask.convertTo(skinMask, CV_8U, 1.0 / 255);

    cv::findNonZero(skinMask, location);

    double max_part_f, x_min_part, x_max_part;

    for (auto &i : location)
    {
      x.push_back(i.x);
      y.push_back(i.y);
    }

    cv::minMaxLoc(y, NULL, &max_part_f);
    cv::minMaxLoc(x, &x_min_part, &x_max_part);

    cv::Mat new_skin_mask = cv::Mat::zeros(skinMask.size(), CV_8U);

    new_skin_mask(cv::Range(face_box_min_y, max_part_f), cv::Range(x_min_part, x_max_part)) =
        skinMask(cv::Range(face_box_min_y, max_part_f), cv::Range(x_min_part, x_max_part));

    return new_skin_mask;
  }

  absl::Status SmoothFaceCalculator::SmoothFace(CalculatorContext *cc,
                                                ImageFormat::Format *target_format,
                                                const std::unordered_map<std::string, cv::Mat> &mask_vec,
                                                const std::tuple<double, double, double, double> &face_box)
  {
    cv::Mat mouth_mask, mouth;

    cv::Mat not_full_face = mask_vec.find("FACE_OVAL")->second.clone() +
                            predict_forehead_mask(mask_vec, std::get<1>(face_box)) -
                            mask_vec.find("LEFT_EYE")->second.clone() -
                            mask_vec.find("RIGHT_EYE")->second.clone() -
                            mask_vec.find("LEFT_BROW")->second.clone() -
                            mask_vec.find("RIGHT_BROW")->second.clone() -
                            mask_vec.find("LIPS")->second.clone();

    cv::resize(not_full_face,
               not_full_face,
               mat_image_.size(), 0, 0,
               cv::INTER_LINEAR);

    std::vector<int> x, y;
    std::vector<cv::Point> location;

    cv::findNonZero(not_full_face, location);

    double min_y, min_x, max_x, max_y;

    for (auto &i : location)
    {
      x.push_back(i.x);
      y.push_back(i.y);
    }

    cv::minMaxLoc(x, &min_x, &max_x);
    cv::minMaxLoc(y, &min_y, &max_y);

    cv::Mat patch_face = mat_image_(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
    cv::Mat patch_nff = not_full_face(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
    cv::Mat patch_new, patch_wow;
    cv::cvtColor(patch_face, patch_wow, cv::COLOR_RGBA2RGB);
    cv::bilateralFilter(patch_wow, patch_new, 12, 50, 50);

    cv::Mat patch_new_nff, patch_new_mask, patch, patch_face_nff;

    patch_new.copyTo(patch_new_nff, patch_nff);

    patch_face.copyTo(patch_face_nff, patch_nff);
    cv::cvtColor(patch_face_nff, patch_face_nff, cv::COLOR_RGBA2RGB);

    patch_new_mask = 0.85 * patch_new_nff + 0.15 * patch_face_nff;

    patch = cv::min(255, patch_new_mask);

    cv::cvtColor(patch, patch, cv::COLOR_RGB2RGBA);

    patch.copyTo(patch_face, patch_nff);

    return absl::OkStatus();
  }

} // namespace mediapipe
