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
//#include <android/log.h>

#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"

namespace mediapipe
{
  namespace
  {
    static const std::vector<cv::Point2f> FFHQ_NORM_LM = {
        {638.68525475 / 1024, 486.24604922 / 1024},
        {389.31496114 / 1024, 485.8921848 / 1024},
        {513.67979275 / 1024, 620.8915371 / 1024},
        {405.50932642 / 1024, 756.52797927 / 1024},
        {622.55630397 / 1024, 756.15509499 / 1024}};

    constexpr char kImageFrameTag[] = "IMAGE";
    constexpr char kVectorTag[] = "VECTOR";
    constexpr char kLandmarksTag[] = "LANDMARKS";
    constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";

    std::tuple<int, int> _normalized_to_pixel_coordinates(float normalized_x,
                                                          float normalized_y, int image_width, int image_height)
    {
      // Converts normalized value pair to pixel coordinates
      int x_px = std::min<int>(floor(normalized_x * image_width), image_width - 1);
      int y_px = std::min<int>(floor(normalized_y * image_height), image_height - 1);

      return {x_px, y_px};
    };

    static const std::vector<cv::Point> FACEMESH_FACE_OVAL{
        {10, 338}, {338, 297}, {297, 332}, {332, 284}, {284, 251}, {251, 389}, {389, 356}, {356, 454}, {454, 323}, {323, 361}, {361, 288}, {288, 397}, {397, 365}, {365, 379}, {379, 378}, {378, 400}, {400, 377}, {377, 152}, {152, 148}, {148, 176}, {176, 149}, {149, 150}, {150, 136}, {136, 172}, {172, 58}, {58, 132}, {132, 93}, {93, 234}, {234, 127}, {127, 162}, {162, 21}, {21, 54}, {54, 103}, {103, 67}, {67, 109}, {109, 10}};

    enum
    {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
      NUM_ATTRIBUTES
    };

    // Round up n to next multiple of m.
    size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; } // NOLINT
    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }

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

    std::tuple<float, cv::Mat, cv::Mat> LandmarkTransform(
        cv::Mat &source,
        cv::Mat &target, float eps = 1e-7)
    {
      cv::Mat source_mean_mat, target_mean_mat, source1ch, target1ch;

      cv::reduce(source, source_mean_mat, 0, CV_REDUCE_AVG, CV_32F);
      cv::reduce(target, target_mean_mat, 0, CV_REDUCE_AVG, CV_32F);

      source -= {source_mean_mat.at<float>(0, 0), source_mean_mat.at<float>(0, 1)};
      target -= {target_mean_mat.at<float>(0, 0), target_mean_mat.at<float>(0, 1)};

      source1ch = source.reshape(1, 5);
      target1ch = target.reshape(1, 5);

      cv::Mat source_std_mat, target_std_mat;
      cv::meanStdDev(source1ch, cv::noArray(), source_std_mat);
      cv::meanStdDev(target1ch, cv::noArray(), target_std_mat);
      source_std_mat.convertTo(source_std_mat, CV_32F);
      target_std_mat.convertTo(target_std_mat, CV_32F);

      float source_std = source_std_mat.at<float>(0, 0);
      float target_std = target_std_mat.at<float>(0, 0);

      source /= source_std + eps;
      target /= target_std + eps;

      cv::Mat u, vt, rotation, w;

      source1ch = source.reshape(1, 5);
      target1ch = target.reshape(1, 5);

      //std::cout << "R (numpy)   = " << std::endl << cv::format(source, cv::Formatter::FMT_NUMPY) << std::endl << std::endl;

      cv::SVD::compute(source1ch.t() * target1ch, w, u, vt);

      rotation = (u * vt).t();

      float scale = target_std / source_std + eps;
      cv::Mat translation;

      cv::subtract(target_mean_mat.reshape(1, 2), scale * rotation * source_mean_mat.reshape(1, 2), translation);

      return std::make_tuple(scale, rotation, translation);
    }

    std::tuple<float, float, float, float> Crop(
        std::unique_ptr<cv::Mat> &image_mat,
        std::tuple<float, float, float, float> roi, float extend = 1.0,
        bool square = false, float shift_x = 0.0, float shift_y = 0.0)
    {
      cv::Mat image = *image_mat.get();

      int width = image_mat->cols;
      int height = image_mat->rows;

      auto &[left, top, right, bottom] = roi;
      int y = static_cast<int>((bottom + top) / 2);
      int x = static_cast<int>((right + left) / 2);

      int size_y = static_cast<int>(extend * (bottom - top) / 2);
      int size_x = static_cast<int>(extend * (right - left) / 2);

      if (square)
        size_x = size_y = std::max(size_x, size_y);

      x += static_cast<int>(shift_x * size_x);
      y += static_cast<int>(shift_y * size_y);

      roi = std::make_tuple(
          std::max(0, x - size_x),
          std::max(0, y - size_y),
          std::min(x + size_x, width),
          std::min(y + size_y, height));

      image = image(cv::Range(bottom, top), cv::Range(left, right));

      if (square)
        cv::copyMakeBorder(
            image, image, std::abs(std::min(0, y - size_y)),
            std::abs(std::min(0, height - y - size_y)),
            std::abs(std::min(0, x - size_x)),
            std::abs(std::min(0, width - x - size_x)),
            cv::BORDER_CONSTANT);

      return roi;
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
                      ImageFormat::Format &target_format,
                      std::vector<std::vector<cv::Point2f>> &lms_out);

    absl::Status Align(std::unique_ptr<cv::Mat> &image_mat,
                       cv::Mat source_lm,
                       cv::Mat target_lm = cv::Mat(FFHQ_NORM_LM), cv::Size size = cv::Size(256, 256),
                       float extend = NULL, std::tuple<float, float, float, float> roi = {NULL, NULL, NULL, NULL});

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    std::vector<std::pair<std::string, const std::vector<int>>> index_dict = {
        {"leftEye", {384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382}},
        {"rightEye", {160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159}},
        {"nose", {4}},
        //{"lips", {0, 13, 14, 17, 84}},
        {"leftLips", {61, 146}},
        {"rightLips", {291, 375}},
    };

    std::unique_ptr<cv::Mat> image_mat;
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

    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ||
              cc->Inputs().HasTag(kNormLandmarksTag))
        << "None of the input streams are provided.";
    RET_CHECK(!(cc->Inputs().HasTag(kLandmarksTag) &&
                cc->Inputs().HasTag(kNormLandmarksTag)))
        << "Can only one type of landmark can be taken. Either absolute or "
           "normalized landmarks.";

    if (cc->Inputs().HasTag(kLandmarksTag))
    {
      cc->Inputs().Tag(kLandmarksTag).Set<std::vector<LandmarkList>>();
    }
    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      cc->Inputs().Tag(kNormLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
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

    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;
    std::vector<std::vector<cv::Point2f>> lms_out;

    MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));

    MP_RETURN_IF_ERROR(Call(cc, image_mat, target_format, lms_out));

    cv::Mat source_lm = cv::Mat(lms_out[0]);

    MP_RETURN_IF_ERROR(Align(image_mat, source_lm));

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
                                         ImageFormat::Format &target_format,
                                         std::vector<std::vector<cv::Point2f>> &lms_out)
  {
    cv::Mat mat_image_ = *image_mat.get();

    int image_width_ = image_mat->cols;
    int image_height_ = image_mat->rows;

    std::vector<cv::Point2f> kps, landmarks;

    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      const std::vector<NormalizedLandmarkList> &landmarkslist =
          cc->Inputs().Tag(kNormLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();

      std::vector<cv::Point2f> point_array;
      for (const auto &face : landmarkslist)
      { 
        for (const auto &[key, value] : index_dict)
        {
          for (auto order : value)
          {

            const NormalizedLandmark &landmark = face.landmark(order);

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
            kps.push_back(cv::Point2f(x, y));
          }

          cv::Mat mean;
          cv::reduce(kps, mean, 1, CV_REDUCE_AVG, CV_32F);
         
          landmarks.push_back({mean.at<float>(0, 0), mean.at<float>(0, 1)});
          
          kps.clear();
        }
        lms_out.push_back(landmarks);
    
        landmarks.clear();
      }
    }

    return absl::OkStatus();
  }

  absl::Status FastUtilsCalculator::Align(std::unique_ptr<cv::Mat> &image_mat,
                                          cv::Mat source_lm,
                                          cv::Mat target_lm, cv::Size size,
                                          float extend, std::tuple<float, float, float, float> roi)
  {
    cv::Mat mat_image_ = *image_mat.get();

    cv::Mat source, target;
    source_lm.convertTo(source, CV_32F);
    target_lm.convertTo(target, CV_32F);

    if (target.at<float>(0, 0) < 1)
    {
      target *= size.width;
    }

    if (std::get<0>(roi) != NULL)
    {
      roi = Crop(image_mat, roi, extend);

      auto [left, top, right, bottom] = roi;
      source(cv::Range(cv::Range::all()), cv::Range(0, 1)) -= left;
      source(cv::Range(cv::Range::all()), cv::Range(1, 2)) -= top;
    }
    auto [scale, rotation, translation] = LandmarkTransform(source, target);

    std::vector<cv::Mat> vec_mat;

    vec_mat.push_back(scale * rotation);
    vec_mat.push_back(translation.reshape(1, {2, 1}));

    cv::Mat transform, image;
    cv::hconcat(vec_mat, transform);
    
    cv::warpAffine(mat_image_, *image_mat, transform, size, 1, 0, 0.0);

    return absl::OkStatus();
  }
} // namespace mediapipe
