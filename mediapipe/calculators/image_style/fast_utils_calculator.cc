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
    static const std::vector<cv::Point> FFHQ_NORM_LM = {
        {638.68525475 / 1024, 486.24604922 / 1024},
        {389.31496114 / 1024, 485.8921848 / 1024},
        {513.67979275 / 1024, 620.8915371 / 1024},
        {405.50932642 / 1024, 756.52797927 / 1024},
        {622.55630397 / 1024, 756.15509499 / 1024}};

    constexpr char kImageFrameTag[] = "IMAGE";
    constexpr char kVectorTag[] = "VECTOR";

    std::tuple<int, int> _normalized_to_pixel_coordinates(float normalized_x,
                                                          float normalized_y, int image_width, int image_height)
    {
      // Converts normalized value pair to pixel coordinates
      int x_px = std::min<int>(floor(normalized_x * image_width), image_width - 1);
      int y_px = std::min<int>(floor(normalized_y * image_height), image_height - 1);

      return {x_px, y_px};
    };

    static const std::unordered_set<cv::Point> FACEMESH_FACE_OVAL =
        {{10, 338}, {338, 297}, {297, 332}, {332, 284}, {284, 251}, {251, 389}, {389, 356}, {356, 454}, {454, 323}, {323, 361}, {361, 288}, {288, 397}, {397, 365}, {365, 379}, {379, 378}, {378, 400}, {400, 377}, {377, 152}, {152, 148}, {148, 176}, {176, 149}, {149, 150}, {150, 136}, {136, 172}, {172, 58}, {58, 132}, {132, 93}, {93, 234}, {234, 127}, {127, 162}, {162, 21}, {21, 54}, {54, 103}, {103, 67}, {67, 109}, {109, 10}};

    enum
    {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
      NUM_ATTRIBUTES
    };

    // Round up n to next multiple of m.
    size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; } // NOLINT
    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }

    using Point = RenderAnnotation::Point;

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

  class FastUtilsCalculator : public CalculatorBase
  {
  public:
    FastUtilsCalculator() = default;
    ~FastUtilsCalculator() override = default;

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

    absl::Status Call(CalculatorContext *cc,
                      std::unique_ptr<cv::Mat> &image_mat,
                      ImageFormat::Format *target_format,
                      const RenderData &render_data,
                      std::unordered_map<std::string, cv::Mat> &all_masks);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    std::unordered_map<std::string, const std::vector<int>> index_dict = {
        {"leftEye", {384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382}},
        {"rightEye", {160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159}},
        {"nose", {4}},
        {"lips", {0, 13, 14, 17, 84}},
        {"leftLips", {61, 146}},
        {"rightLips", {291, 375}},
    };

    int width_ = 0;
    int height_ = 0;
    int width_canvas_ = 0; // Size of overlay drawing texture canvas.
    int height_canvas_ = 0;

    int max_num_faces = 1;
    bool refine_landmarks = True;
    double min_detection_confidence = 0.5;
    double min_tracking_confidence = 0.5;
  };
  REGISTER_CALCULATOR(FastUtilsCalculator);

  absl::Status FastUtilsCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      CHECK(cc->Outputs().HasTag(kImageFrameTag));
    }

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status FastUtilsCalculator::Open(CalculatorContext *cc)
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

  absl::Status FastUtilsCalculator::Process(CalculatorContext *cc)
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

    if (cc->Outputs().HasTag(kImageFrameTag))
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
        MP_RETURN_IF_ERROR(Call(cc, image_mat, &target_format, render_data, all_masks));
      }
      else
      {
        RET_CHECK_EQ(kVectorTag, tag);
        const std::vector<RenderData> &render_data_vec =
            cc->Inputs().Get(id).Get<std::vector<RenderData>>();
        for (const RenderData &render_data : render_data_vec)
        {
          MP_RETURN_IF_ERROR(Call(cc, image_mat, &target_format, render_data, all_masks));
        }
      }
    }

    // Copy the rendered image to output.
    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr, image_mat));

    return absl::OkStatus();
  }

  absl::Status FastUtilsCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status FastUtilsCalculator::RenderToCpu(
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

  absl::Status FastUtilsCalculator::CreateRenderTargetCpu(
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

  absl::Status FastUtilsCalculator::Call(CalculatorContext *cc,
                                         std::unique_ptr<cv::Mat> &image_mat,
                                         ImageFormat::Format *target_format,
                                         const RenderData &render_data,
                                         std::unordered_map<std::string, cv::Mat> &all_masks)
  {
    cv::Mat mat_image_ = *image_mat.get();

    int image_width_ = image_mat->cols;
    int image_height_ = image_mat->rows;

    cv::Mat mask;
    std::vector<cv::Point> kps, landmarks;
    std::vector<std::vector<cv::Point>> lms_out;
    
    int c = 0;
    
    for (const auto &[key, value] : index_dict)
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
              kps.push_back(cv::Point(x, y));
            }
            c += 1;
          }
        }
      }
      double sumx = 0, sumy = 0, meanx, meany;

      for (auto p : kps)
      {
        sumx += p.x;
        sumy += p.y;
      }
      meanx = sumx / kps.size();
      meany = sumy / kps.size();

      landmarks.push_back({meanx, meany});

      kps.clear();
    }

    lms_out.push_back(landmarks);

    return absl::OkStatus();
  }

} // namespace mediapipe
