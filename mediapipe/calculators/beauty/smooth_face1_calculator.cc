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
#include "mediapipe/framework/calculator_options.pb.h"
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
    constexpr char kFaceTag[] = "FACE";

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
  } // namespace

  class SmoothFaceCalculator1 : public CalculatorBase
  {
  public:
    SmoothFaceCalculator1() = default;
    ~SmoothFaceCalculator1() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status RenderToCpu(
        CalculatorContext *cc);

    absl::Status SmoothFace(CalculatorContext *cc,
                            const std::unordered_map<std::string, cv::Mat> &mask_vec,
                            const std::tuple<double, double, double, double> &face_box);

    cv::Mat predict_forehead_mask(const std::unordered_map<std::string, cv::Mat> &mask_vec, double face_box_min_y);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    int image_width_;
    int image_height_;
    cv::Mat mat_image_;
    cv::Mat not_full_face;
    std::vector<double> face_box;
    std::pair<cv::Mat, std::vector<double>> face;
  };
  REGISTER_CALCULATOR(SmoothFaceCalculator1);

  absl::Status SmoothFaceCalculator1::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kMatTag))
    {
      cc->Inputs().Tag(kMatTag).Set<cv::Mat>();
      CHECK(cc->Outputs().HasTag(kMatTag));
      CHECK(cc->Outputs().HasTag(kMaskTag));
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

      if (tag == kFaceTag)
      {
        cc->Inputs().Get(id).Set<std::vector<std::tuple<double, double, double, double>>>();
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
    if (cc->Outputs().HasTag(kFaceTag))
    {
      cc->Outputs().Tag(kFaceTag).Set<std::pair<cv::Mat, std::vector<double>>>();
    }

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator1::Open(CalculatorContext *cc)
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

  absl::Status SmoothFaceCalculator1::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kMatTag) &&
        cc->Inputs().Tag(kMatTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;

    const cv::Mat &input_mat =
        cc->Inputs().Tag(kMatTag).Get<cv::Mat>();

    mat_image_ = input_mat.clone();

    image_width_ = input_mat.cols;
    image_height_ = input_mat.rows;

    if (cc->Inputs().HasTag(kMaskTag) &&
        !cc->Inputs().Tag(kMaskTag).IsEmpty() &&
        cc->Inputs().HasTag(kFaceTag) &&
        !cc->Inputs().Tag(kFaceTag).IsEmpty())
    {
      const std::vector<std::unordered_map<std::string, cv::Mat>> &mask_vec =
          cc->Inputs().Tag(kMaskTag).Get<std::vector<std::unordered_map<std::string, cv::Mat>>>();

      const std::vector<std::tuple<double, double, double, double>> &face_boxes =
          cc->Inputs().Tag(kFaceTag).Get<std::vector<std::tuple<double, double, double, double>>>();

      if (mask_vec.size() > 0 && face_boxes.size() > 0)
      {
        for (int i = 0; i < mask_vec.size(); i++)
          MP_RETURN_IF_ERROR(SmoothFace(cc, mask_vec[i], face_boxes[i]));
      }
    }

    MP_RETURN_IF_ERROR(RenderToCpu(cc));

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator1::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator1::RenderToCpu(
      CalculatorContext *cc)
  {
    auto output_frame1 = absl::make_unique<cv::Mat>(mat_image_);

    if (cc->Outputs().HasTag(kMatTag))
    {
      cc->Outputs()
          .Tag(kMatTag)
          .Add(output_frame1.release(), cc->InputTimestamp());
    }

    not_full_face.convertTo(not_full_face, CV_32F, 1.0 / 255);
    auto output_frame2 = absl::make_unique<cv::Mat>(not_full_face);

    if (cc->Outputs().HasTag(kMaskTag))
    {
      cc->Outputs()
          .Tag(kMaskTag)
          .Add(output_frame2.release(), cc->InputTimestamp());
    }

    auto output_frame3 = absl::make_unique<std::pair<cv::Mat, std::vector<double>>>(
        face);

    if (cc->Outputs().HasTag(kFaceTag))
    {
      cc->Outputs()
          .Tag(kFaceTag)
          .Add(output_frame3.release(), cc->InputTimestamp());
    }
    return absl::OkStatus();
  }

  cv::Mat SmoothFaceCalculator1::predict_forehead_mask(const std::unordered_map<std::string, cv::Mat> &mask_vec, double face_box_min_y)
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

  absl::Status SmoothFaceCalculator1::SmoothFace(CalculatorContext *cc,
                                                 const std::unordered_map<std::string, cv::Mat> &mask_vec,
                                                 const std::tuple<double, double, double, double> &face_boxx)
  {
    not_full_face = mask_vec.find("FACE_OVAL")->second.clone() -
                    //                    predict_forehead_mask(mask_vec, std::get<1>(face_boxx)) -
                    mask_vec.find("LEFT_EYE")->second.clone() -
                    mask_vec.find("RIGHT_EYE")->second.clone() -
                    mask_vec.find("LEFT_BROW")->second.clone() -
                    mask_vec.find("RIGHT_BROW")->second.clone() -
                    mask_vec.find("LIPS")->second.clone();

    cv::resize(not_full_face,
               not_full_face,
               mat_image_.size(), 0, 0,
               cv::INTER_LINEAR);

    cv::Rect rect = cv::boundingRect(not_full_face);

    if (!rect.empty())
    {
      double min_y = rect.y, max_y = rect.y + rect.height,
             max_x = rect.x + rect.width, min_x = rect.x;
      face.second.push_back(min_x);
      face.second.push_back(min_y);
      face.second.push_back(max_x);
      face.second.push_back(max_y);
      face.first = mat_image_;
    }
    return absl::OkStatus();
  }

} // namespace mediapipe
