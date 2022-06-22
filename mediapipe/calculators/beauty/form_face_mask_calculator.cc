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
#include <string>
#include <map>
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
#include "mediapipe/util/annotation_renderer.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/color.pb.h"

namespace mediapipe
{
  namespace
  {

    constexpr char kVectorTag[] = "VECTOR";
    constexpr char kMaskTag[] = "MASK";
    constexpr char kFaceBoxTag[] = "FACEBOX";
    constexpr char kImageFrameTag[] = "IMAGE";

    enum
    {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
      NUM_ATTRIBUTES
    };

    // Round up n to next multiple of m.
    size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; } // NOLINT

    // When using GPU, this color will become transparent when the calculator
    // merges the annotation overlay with the image frame. As a result, drawing in
    // this color is not supported and it should be set to something unlikely used.
    constexpr uchar kAnnotationBackgroundColor = 2; // Grayscale value.

    // Future Image type.
    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }

    static const std::vector<int> UPPER_LIP = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78};
    static const std::vector<int> LOWER_LIP = {61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146};
    static const std::vector<int> FACE_OVAL = {10, 338, 338, 297, 297, 332, 332, 284, 284, 251, 251, 389, 389, 356, 356,
                                               454, 454, 323, 323, 361, 361, 288, 288, 397, 397, 365, 365, 379, 379, 378,
                                               378, 400, 400, 377, 377, 152, 152, 148, 148, 176, 176, 149, 149, 150, 150,
                                               136, 136, 172, 172, 58, 58, 132, 132, 93, 93, 234, 234, 127, 127, 162, 162,
                                               21, 21, 54, 54, 103, 103, 67, 67, 109, 109, 10};
    static const std::vector<int> MOUTH_INSIDE = {78, 191, 80, 81, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95};
    static const std::vector<int> PART_FOREHEAD_B = {21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 301, 293, 334, 296, 336, 9, 107, 66, 105, 63, 71};
    static const std::vector<int> LEFT_EYE = {130, 33, 246, 161, 160, 159, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7};
    static const std::vector<int> RIGHT_EYE = {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382};
    static const std::vector<int> LIPS = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146};
    static const std::vector<int> LEFT_BROW = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46};
    static const std::vector<int> RIGHT_BROW = {336, 296, 334, 293, 301, 300, 283, 282, 295, 285};

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

  } // namespace

  class FormFaceMaskCalculator : public CalculatorBase
  {
  public:
    FormFaceMaskCalculator() = default;
    ~FormFaceMaskCalculator() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status CreateRenderTargetCpu(CalculatorContext *cc,
                                       std::unique_ptr<cv::Mat> &image_mat,
                                       ImageFormat::Format *target_format);

    absl::Status RenderToCpu(CalculatorContext *cc, std::unordered_map<std::string, cv::Mat> &all_masks);

    absl::Status FormFacePartMask(CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat,
                                  ImageFormat::Format *target_format,
                                  const RenderData &render_data,
                                  std::unordered_map<std::string, cv::Mat> &all_masks);

    absl::Status GetFaceBox(std::unique_ptr<cv::Mat> &image_mat,
                            const RenderData &render_data);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    int width_ = 0;
    int height_ = 0;
    int width_canvas_ = 0; // Size of overlay drawing texture canvas.
    int height_canvas_ = 0;
    float scale_factor_ = 1.0;
    std::tuple<double, double, double, double> face_box;
  };
  REGISTER_CALCULATOR(FormFaceMaskCalculator);

  absl::Status FormFaceMaskCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      CHECK(cc->Outputs().HasTag(kMaskTag));
    }

    // Data streams to render.
    for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
         ++id)
    {
      auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
      std::string tag = tag_and_index.first;
      if (tag == kVectorTag)
      {
        cc->Inputs().Get(id).Set<std::vector<RenderData>>();
      }
      else if (tag.empty())
      {
        // Empty tag defaults to accepting a single object of RenderData type.
        cc->Inputs().Get(id).Set<RenderData>();
      }
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

  absl::Status FormFaceMaskCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    return absl::OkStatus();
  }

  absl::Status FormFaceMaskCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.
    std::unique_ptr<cv::Mat> image_mat;
    ImageFormat::Format target_format;
    std::unordered_map<std::string, cv::Mat> all_masks;

    if (cc->Outputs().HasTag(kMaskTag))
    {
      MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));
    }

    // Render streams onto render target.
    for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
         ++id)
    {
      auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
      std::string tag = tag_and_index.first;
      if (!tag.empty() && tag != kVectorTag)
      {
        continue;
      }
      if (cc->Inputs().Get(id).IsEmpty())
      {
        continue;
      }
      if (tag.empty())
      {
        // Empty tag defaults to accepting a single object of RenderData type.
        const RenderData &render_data = cc->Inputs().Get(id).Get<RenderData>();
        MP_RETURN_IF_ERROR(FormFacePartMask(cc, image_mat, &target_format, render_data, all_masks));

        if (cc->Outputs().HasTag(kFaceBoxTag))
        {
          MP_RETURN_IF_ERROR(GetFaceBox(image_mat, render_data));
        }
      }
      else
      {
        RET_CHECK_EQ(kVectorTag, tag);
        const std::vector<RenderData> &render_data_vec =
            cc->Inputs().Get(id).Get<std::vector<RenderData>>();
        for (const RenderData &render_data : render_data_vec)
        {
          MP_RETURN_IF_ERROR(FormFacePartMask(cc, image_mat, &target_format, render_data, all_masks));
        }
      }
    }

    // Copy the rendered image to output.
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, all_masks));

    return absl::OkStatus();
  }

  absl::Status FormFaceMaskCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status FormFaceMaskCalculator::RenderToCpu(CalculatorContext *cc,
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

  absl::Status FormFaceMaskCalculator::CreateRenderTargetCpu(
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
          150, 150, CV_8UC3,
          cv::Scalar(255, 255,
                     255));
      *target_format = ImageFormat::SRGB;
    }

    return absl::OkStatus();
  }

  absl::Status FormFaceMaskCalculator::GetFaceBox(std::unique_ptr<cv::Mat> &image_mat,
                                                  const RenderData &render_data)
  {
    cv::Mat mat_image_ = *image_mat.get();

    int image_width_ = image_mat->cols;
    int image_height_ = image_mat->rows;

    std::vector<int> x_s, y_s;
    double box_min_y, box_max_y, box_max_x, box_min_x;

    for (auto &annotation : render_data.render_annotations())
    {
      if (annotation.data_case() == RenderAnnotation::kPoint)
      {
        const auto &point = annotation.point();
        int x = -1;
        int y = -1;
        if (point.normalized())
        {
          CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                             image_height_, &x, &y));
        }
        else
        {
          x = static_cast<int>(point.x() * scale_factor_);
          y = static_cast<int>(point.y() * scale_factor_);
        }
        x_s.push_back(point.x());
        x_s.push_back(point.y());
      }
    }
    cv::minMaxLoc(y_s, &box_min_y, &box_max_y);
    cv::minMaxLoc(x_s, &box_min_x, &box_max_x);
    box_min_y = box_min_y * 0.9;
    face_box = std::make_tuple(box_min_x, box_min_y, box_max_x, box_max_y);

    return absl::OkStatus();
  }

  absl::Status FormFaceMaskCalculator::FormFacePartMask(CalculatorContext *cc,
                                                        std::unique_ptr<cv::Mat> &image_mat,
                                                        ImageFormat::Format *target_format,
                                                        const RenderData &render_data,
                                                        std::unordered_map<std::string, cv::Mat> &all_masks)
  {
    cv::Mat mat_image_ = *image_mat.get();

    int image_width_ = image_mat->cols;
    int image_height_ = image_mat->rows;

    std::unordered_map<std::string, const std::vector<int>> orderList;
    orderList.insert(make_pair("UPPER_LIP", UPPER_LIP));
    orderList.insert(make_pair("LOWER_LIP", LOWER_LIP));
    orderList.insert(make_pair("FACE_OVAL", FACE_OVAL));
    orderList.insert(make_pair("MOUTH_INSIDE", MOUTH_INSIDE));
    orderList.insert(make_pair("LEFT_EYE", LEFT_EYE));
    orderList.insert(make_pair("RIGHT_EYE", RIGHT_EYE));
    orderList.insert(make_pair("LEFT_BROW", LEFT_BROW));
    orderList.insert(make_pair("RIGHT_BROW", RIGHT_BROW));
    orderList.insert(make_pair("LIPS", LIPS));
    orderList.insert(make_pair("PART_FOREHEAD_B", PART_FOREHEAD_B));

    cv::Mat mask;
    std::vector<cv::Point> point_array;
    int c = 0;
    for (const auto &[key, value] : orderList)
    {
      for (auto order : value)
      {
        c = 0;
        for (auto &annotation : render_data.render_annotations())
        {
          if (annotation.data_case() == RenderAnnotation::kPoint)
          {
            if (order == c)
            {
              const auto &point = annotation.point();
              int x = -1;
              int y = -1;
              CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                                 image_height_, &x, &y));
              point_array.push_back(cv::Point(x, y));
            }
            c += 1;
          }
        }
      }
      std::vector<std::vector<cv::Point>> point_vec;
      point_vec.push_back(point_array);
      mask = cv::Mat::zeros(mat_image_.size(), CV_32FC1);
      cv::fillPoly(mask, point_vec, cv::Scalar::all(255), cv::LINE_AA);
      mask.convertTo(mask, CV_8U);
      all_masks.insert(make_pair(key, mask));
      point_vec.clear();
      point_array.clear();
    }

    return absl::OkStatus();
  }

} // namespace mediapipe
