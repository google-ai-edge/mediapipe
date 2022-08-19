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
    absl::Status RenderToCpu(
        CalculatorContext *cc);
    absl::Status DrawLipstick(CalculatorContext *cc,
                              std::unordered_map<std::string, cv::Mat> &mask_vec);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;
    std::unordered_map<std::string, cv::Mat> all_masks;
    cv::Mat spec_lips_mask;
    cv::Mat mat_image_;
  };
  REGISTER_CALCULATOR(DrawLipstickCalculator);

  absl::Status DrawLipstickCalculator::GetContract(CalculatorContract *cc)
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

  absl::Status DrawLipstickCalculator::Open(CalculatorContext *cc)
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

  absl::Status DrawLipstickCalculator::Process(CalculatorContext *cc)
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

    if (cc->Inputs().HasTag(kMaskTag) &&
        !cc->Inputs().Tag(kMaskTag).IsEmpty())
    {
      const std::vector<std::unordered_map<std::string, cv::Mat>> &mask_vec =
          cc->Inputs().Tag(kMaskTag).Get<std::vector<std::unordered_map<std::string, cv::Mat>>>();

      if (mask_vec.size() > 0)
      {
        for (auto mask : mask_vec){
          MP_RETURN_IF_ERROR(DrawLipstick(cc, mask));
        }
      }
    }

    MP_RETURN_IF_ERROR(RenderToCpu(cc));

    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::RenderToCpu(
      CalculatorContext *cc)
  {
    auto output_frame = absl::make_unique<cv::Mat>(mat_image_);
 
    if (cc->Outputs().HasTag(kMatTag))
    {
      cc->Outputs()
          .Tag(kMatTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    spec_lips_mask.convertTo(spec_lips_mask, CV_32F, 1.0 / 255);
    auto output_frame2 = absl::make_unique<cv::Mat>(spec_lips_mask);

    if (cc->Outputs().HasTag(kMaskTag))
    {
      cc->Outputs()
          .Tag(kMaskTag)
          .Add(output_frame2.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status DrawLipstickCalculator::DrawLipstick(CalculatorContext *cc,
                                                     std::unordered_map<std::string, cv::Mat> &mask_vec)
  {
    cv::Mat upper_lips_mask = mask_vec.find("UPPER_LIP")->second;
    cv::Mat lower_lips_mask = mask_vec.find("LOWER_LIP")->second;

    spec_lips_mask = upper_lips_mask + lower_lips_mask;

    cv::resize(spec_lips_mask, spec_lips_mask, mat_image_.size(), cv::INTER_LINEAR);

    cv::Rect rect = cv::boundingRect(spec_lips_mask);

    if (!rect.empty())
    {
      double min_y = rect.y, max_y = rect.y + rect.height,
             max_x = rect.x + rect.width, min_x = rect.x;

      cv::Mat lips_crop_mask = spec_lips_mask(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
      lips_crop_mask.convertTo(lips_crop_mask, CV_32F, 1.0 / 255);

      cv::Mat lips_crop = cv::Mat(mat_image_(cv::Range(min_y, max_y), cv::Range(min_x, max_x)));
      cv::Mat lips_blend = cv::Mat(lips_crop.size().height, lips_crop.size().width, CV_32FC4, cv::Scalar(255.0, 0, 0, 0));
      std::vector<cv::Mat> channels(4);

      cv::split(lips_blend, channels);
      channels[3] = lips_crop_mask * 20;
      cv::merge(channels, lips_blend);

      cv::Mat tmp_lip_mask;
      channels[3].convertTo(tmp_lip_mask, CV_32F, 1.0 / 255);

      cv::merge(std::vector{tmp_lip_mask, tmp_lip_mask, tmp_lip_mask, tmp_lip_mask}, tmp_lip_mask);
      cv::multiply(lips_blend, tmp_lip_mask, lips_blend, 1.0, CV_32F);
      cv::subtract(1.0, tmp_lip_mask, tmp_lip_mask, cv::noArray(), CV_32F);
      cv::multiply(lips_crop, tmp_lip_mask, lips_crop, 1.0, CV_8U);
      cv::add(lips_blend, lips_crop, lips_crop, cv::noArray(), CV_8U);
      lips_crop = cv::abs(lips_crop);

      cvtColor(lips_crop, lips_crop, cv::COLOR_RGBA2RGB);

      cv::Mat slice = mat_image_(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
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
