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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"

using namespace std;
namespace mediapipe
{
  namespace api2
  {
    class ImageFrameToMatCalculator : public Node
    {
    public:
      static constexpr Input<ImageFrame> kImageFrame{"IMAGE"};
      static constexpr Output<cv::Mat> kOut{"MAT"};

      MEDIAPIPE_NODE_CONTRACT(kImageFrame, kOut);

      static absl::Status UpdateContract(CalculatorContract *cc)
      {
        RET_CHECK(kOut(cc).IsConnected())
            << "At least one output stream is expected.";
        return absl::OkStatus();
      }

      absl::Status Process(CalculatorContext *cc)
      {
        // Initialize render target, drawn with OpenCV.
        std::unique_ptr<cv::Mat> image_mat;
        ImageFormat::Format target_format;

        MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));

        MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat));

        return absl::OkStatus();
      }

      absl::Status Close(CalculatorContext *cc)
      {
        return absl::OkStatus();
      }

      absl::Status RenderToCpu(
          CalculatorContext *cc, const ImageFormat::Format &target_format, std::unique_ptr<cv::Mat> &image_mat)
      {
        auto output_frame = absl::make_unique<cv::Mat>(*image_mat);

        kOut(cc).Send(std::move(output_frame));

        return absl::OkStatus();
      }

      absl::Status CreateRenderTargetCpu(
          CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat,
          ImageFormat::Format *target_format)
      {
        if (!kImageFrame(cc).IsEmpty())
        {
          const auto &input_frame = *kImageFrame(cc);

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
              1920, 1080, CV_8UC4,
              cv::Scalar(cv::Scalar::all(255)));
          *target_format = ImageFormat::SRGBA;
        }

        return absl::OkStatus();
      }
    };
    MEDIAPIPE_REGISTER_NODE(ImageFrameToMatCalculator); // namespace mediapipe
  }
}
