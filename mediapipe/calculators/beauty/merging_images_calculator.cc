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
    constexpr char kImageFrameTag[] = "IMAGE";
  } // namespace

  class MergeImagesCalculator : public CalculatorBase
  {
  public:
    MergeImagesCalculator() = default;
    ~MergeImagesCalculator() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status SmoothEnd(CalculatorContext *cc,
                            const std::vector<double> &face_box);

    cv::Mat mat_image_;
  };
  REGISTER_CALCULATOR(MergeImagesCalculator);

  absl::Status MergeImagesCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);
    CHECK(cc->Outputs().HasTag(kImageFrameTag));

    // Data streams to render.
    for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
         ++id)
    {
      auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
      std::string tag = tag_and_index.first;
      if (tag == kMatTag)
      {
        cc->Inputs().Get(id).Set<cv::Mat>();
      }
      else
      {
        // Every other tag defaults to accepting a single object of Mat type.
        cc->Inputs().Get(id).Set<cv::Mat>();
      }
    }

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status MergeImagesCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    // Set the output header based on the input header (if present).
    const char *tag = kMatTag;
    if (!cc->Inputs().Tag(tag).Header().IsEmpty())
    {
      const auto &input_header =
          cc->Inputs().Tag(tag).Header().Get<VideoHeader>();
      auto *output_video_header = new VideoHeader(input_header);
      cc->Outputs().Tag(tag).SetHeader(Adopt(output_video_header));
    }

    return absl::OkStatus();
  }

  absl::Status MergeImagesCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kMatTag) &&
        cc->Inputs().Tag(kMatTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    const auto &input_mat_c = cc->Inputs().Get(kMatTag, 0).Get<cv::Mat>();
    cv::Mat input_mat = input_mat_c.clone();
    cv::Mat input_mat2 = input_mat.clone();
    cv::Mat mask = cc->Inputs().Get(kMaskTag, 0).Get<cv::Mat>();

    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGBA);
     
    input_mat.convertTo(input_mat, CV_32F);
    mat_image_ = mask.mul(input_mat);
    int image_number = cc->Inputs().NumEntries() / 2;
    cv::Mat all_masks = mask;

    for (int i = 1; i < image_number; i++)
    {
        const auto &input_mat_c = cc->Inputs().Get(kMatTag, i).Get<cv::Mat>();
        cv::Mat input_mat = input_mat_c.clone();
        mask = cc->Inputs().Get(kMaskTag, i).Get<cv::Mat>();
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGBA);
        input_mat.convertTo(input_mat, CV_32F);
        mat_image_ += mask.mul(input_mat);
        all_masks += mask;

    }

    input_mat2.convertTo(input_mat2, CV_32F);
    cv::threshold(all_masks, all_masks, 1, 255, CV_THRESH_TRUNC);
    input_mat2 = input_mat2.mul(1-(all_masks*255));

    input_mat2.convertTo(input_mat2, CV_8U);
    mat_image_.convertTo(mat_image_, CV_8U);

    mat_image_ += input_mat2;

    std::unique_ptr<cv::Mat> image_mat = absl::make_unique<cv::Mat>(
          mat_image_.rows, mat_image_.cols, ImageFormat::SRGBA);
    mat_image_.convertTo(mat_image_, CV_8U);
    mat_image_.copyTo(*image_mat);

    uchar *image_mat_ptr = image_mat->data;
 
    // Copy the rendered image to output.
    auto output_frame = absl::make_unique<ImageFrame>(
        ImageFormat::SRGBA, mat_image_.cols, mat_image_.rows);

    output_frame->CopyPixelData(ImageFormat::SRGBA, mat_image_.cols, mat_image_.rows, image_mat_ptr,
                                ImageFrame::kDefaultAlignmentBoundary);

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs()
          .Tag(kImageFrameTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status MergeImagesCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }
} // namespace mediapipe
