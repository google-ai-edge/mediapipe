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
    constexpr char kFaceBoxTag[] = "FACEBOX";
    constexpr char kMatTag[] = "MAT";
    constexpr char kImageNewTag[] = "IMAGE2";

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
  } // namespace

  class SmoothFaceCalculator2 : public CalculatorBase
  {
  public:
    SmoothFaceCalculator2() = default;
    ~SmoothFaceCalculator2() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status RenderToCpu(
        CalculatorContext *cc);

    absl::Status SmoothEnd(CalculatorContext *cc,
                           const std::vector<double> &face_box);

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    int image_width_;
    int image_height_;
    cv::Mat mat_image_;
    cv::Mat new_image_;
    cv::Mat not_full_face;
  };
  REGISTER_CALCULATOR(SmoothFaceCalculator2);

  absl::Status SmoothFaceCalculator2::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kMatTag))
    {
      cc->Inputs().Tag(kMatTag).Set<cv::Mat>();
      CHECK(cc->Outputs().HasTag(kMatTag));
    }
    if (cc->Inputs().HasTag(kImageNewTag))
    {
      cc->Inputs().Tag(kImageNewTag).Set<cv::Mat>();
    }
    if (cc->Inputs().HasTag(kMaskTag))
    {
      cc->Inputs().Tag(kMaskTag).Set<cv::Mat>();
    }

    if (cc->Inputs().HasTag(kFaceBoxTag))
    {
      cc->Inputs().Tag(kFaceBoxTag).Set<std::pair<cv::Mat, std::vector<double>>>();
    }

    if (cc->Outputs().HasTag(kMatTag))
    {
      cc->Outputs().Tag(kMatTag).Set<cv::Mat>();
    }

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator2::Open(CalculatorContext *cc)
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

  absl::Status SmoothFaceCalculator2::Process(CalculatorContext *cc)
  {

    if (cc->Inputs().HasTag(kMatTag) &&
        cc->Inputs().Tag(kMatTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kImageNewTag) &&
        cc->Inputs().Tag(kImageNewTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kMaskTag) &&
        cc->Inputs().Tag(kMaskTag).IsEmpty())
    {
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kFaceBoxTag) &&
        cc->Inputs().Tag(kFaceBoxTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    not_full_face = cc->Inputs().Tag(kMaskTag).Get<cv::Mat>();
    not_full_face.convertTo(not_full_face, CV_8U, 255);

    const cv::Mat &input_mat =
        cc->Inputs().Tag(kMatTag).Get<cv::Mat>();

    mat_image_ = input_mat.clone();

    const cv::Mat &input_new =
        cc->Inputs().Tag(kImageNewTag).Get<cv::Mat>();

    new_image_ = input_new.clone();

    image_width_ = input_mat.cols;
    image_height_ = input_mat.rows;

    const auto &face_box_pair =
        cc->Inputs().Tag(kFaceBoxTag).Get<std::pair<cv::Mat, std::vector<double>>>();

    const auto &face_box = face_box_pair.second;

    if (!face_box.empty())
    {
      MP_RETURN_IF_ERROR(SmoothEnd(cc, face_box));
    }

    MP_RETURN_IF_ERROR(RenderToCpu(cc));

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator2::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator2::RenderToCpu(
      CalculatorContext *cc)
  {
    auto output_frame = absl::make_unique<cv::Mat>(mat_image_);

    if (cc->Outputs().HasTag(kMatTag))
    {
      cc->Outputs()
          .Tag(kMatTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status SmoothFaceCalculator2::SmoothEnd(CalculatorContext *cc,
                                                const std::vector<double> &face_box)
  {
    cv::Mat patch_face = mat_image_(cv::Range(face_box[1], face_box[3]),
                                    cv::Range(face_box[0], face_box[2]));
    cv::Mat patch_nff = not_full_face(cv::Range(face_box[1], face_box[3]),
                                      cv::Range(face_box[0], face_box[2]));

    cv::Mat patch_new_nff, patch_new_mask, patch, patch_face_nff;

    new_image_.copyTo(patch_new_nff, patch_nff);

    patch_face.copyTo(patch_face_nff, patch_nff);
    cv::cvtColor(patch_face_nff, patch_face_nff, cv::COLOR_RGBA2RGB);
    cv::cvtColor(patch_new_nff, patch_new_nff, cv::COLOR_RGBA2RGB);
    
    patch_new_mask = 0.85 * patch_new_nff + 0.15 * patch_face_nff;

    patch = cv::min(255, patch_new_mask);

    cv::cvtColor(patch, patch, cv::COLOR_RGB2RGBA);

    patch.copyTo(patch_face, patch_nff);

    return absl::OkStatus();
  }

} // namespace mediapipe
