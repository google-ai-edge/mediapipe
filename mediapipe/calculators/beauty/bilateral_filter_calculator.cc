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
    constexpr char kImageFrameTag[] = "IMAGE";
    constexpr char kOutTag[] = "CVMAT";

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
    absl::Status BilateralFilter(CalculatorContext *cc,
                                 const std::vector<double> &face_box);

    absl::Status RenderToCpu(CalculatorContext *cc);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    int image_width_;
    int image_height_;
    cv::Mat mat_image_;
    cv::Mat out_mat;
  };
  REGISTER_CALCULATOR(BilateralCalculator);

  absl::Status BilateralCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<std::pair<cv::Mat, std::vector<double>>>();
      CHECK(cc->Outputs().HasTag(kOutTag));
    }

    if (cc->Outputs().HasTag(kOutTag))
    {
      cc->Outputs().Tag(kOutTag).Set<cv::Mat>();
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
    const char *out_tag = kOutTag;
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

    const std::pair<cv::Mat, std::vector<double>> &face =
        cc->Inputs().Tag(kImageFrameTag).Get<std::pair<cv::Mat, std::vector<double>>>();
    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;
    const std::vector<double> &face_box = face.second;
    mat_image_ = face.first.clone();

    image_width_ = mat_image_.cols;
    image_height_ = mat_image_.rows;

    if (!face_box.empty())
      MP_RETURN_IF_ERROR(BilateralFilter(cc, face_box));

    MP_RETURN_IF_ERROR(RenderToCpu(cc));

    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status BilateralCalculator::BilateralFilter(CalculatorContext *cc,
                                                    const std::vector<double> &face_box)
  {
    cv::Mat patch_wow = mat_image_(cv::Range(face_box[1], face_box[3]),
                                   cv::Range(face_box[0], face_box[2]));

    cv::cvtColor(patch_wow, patch_wow, CV_RGBA2RGB);
    cv::bilateralFilter(patch_wow, out_mat, 12, 50, 50);
    cv::cvtColor(out_mat, out_mat, CV_RGB2RGBA);
    
    return absl::OkStatus();
  }
  absl::Status BilateralCalculator::RenderToCpu(CalculatorContext *cc)
  {
    auto out_mat_ptr = absl::make_unique<cv::Mat>(out_mat);

    if (cc->Outputs().HasTag(kOutTag))
    {
      cc->Outputs()
          .Tag(kOutTag)
          .Add(out_mat_ptr.release(), cc->InputTimestamp());
    }
    return absl::OkStatus();
  }

} // namespace mediapipe
