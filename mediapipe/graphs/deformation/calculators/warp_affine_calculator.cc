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

#include "Tensor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"

using namespace cv;
using namespace std; 

namespace mediapipe
{
    namespace
    {
        constexpr char kImageFrameTag[] = "IMAGE";
        constexpr char kSrcTag[] = "SRC_TENSOR";
        constexpr char kDstTag[] = "DST_TENSOR";

        inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }
    } // namespace

    class WarpAffineCalculator : public CalculatorBase
    {
    public:
        WarpAffineCalculator() = default;
        ~WarpAffineCalculator() override = default;

        static absl::Status GetContract(CalculatorContract *cc);

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

        absl::Status AffineTransform(CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat, Tensor<double> _src, Tensor<double> _dst);

        bool image_frame_available_ = false;
        std::unique_ptr<Mat> image_mat;
    };

    absl::Status WarpAffineCalculator::GetContract(CalculatorContract *cc)
    {
        RET_CHECK(cc->Inputs().HasTag(kImageFrameTag));

        if (cc->Inputs().HasTag(kImageFrameTag))
        {
            cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
        }
        if (cc->Inputs().HasTag(kSrcTag))
        {
            cc->Inputs().Tag(kSrcTag).Set<Tensor<double>>();
        }
        if (cc->Inputs().HasTag(kDstTag))
        {
            cc->Inputs().Tag(kDstTag).Set<Tensor<double>>();
        }
        if (cc->Outputs().HasTag(kImageFrameTag))
        {
            cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
        }

        return absl::OkStatus();
    }

    absl::Status WarpAffineCalculator::Open(CalculatorContext *cc)
    {
        cc->SetOffset(TimestampDiff(0));

        if (cc->Inputs().HasTag(kImageFrameTag) || HasImageTag(cc))
        {
            image_frame_available_ = true;
        }

        return absl::OkStatus();
    }

    absl::Status WarpAffineCalculator::Process(CalculatorContext *cc)
    {
        if (cc->Inputs().Tag(kImageFrameTag).IsEmpty())
        {
            return absl::OkStatus();
        }

        ImageFormat::Format target_format;

        MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));

        if (!cc->Inputs().Tag(kSrcTag).IsEmpty() && !cc->Inputs().Tag(kDstTag).IsEmpty())
        {
            const Tensor<double> _src = cc->Inputs().Tag(kSrcTag).Get<Tensor<double>>();
            const Tensor<double> _dst = cc->Inputs().Tag(kDstTag).Get<Tensor<double>>();
            MP_RETURN_IF_ERROR(AffineTransform(cc, image_mat, _src, _dst));
        }

        // Copy the rendered image to output.
        uchar *image_mat_ptr = image_mat->data;
        MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr, image_mat));

        return absl::OkStatus();
    }

    absl::Status WarpAffineCalculator::Close(CalculatorContext *cc)
    {
        return absl::OkStatus();
    }

    absl::Status WarpAffineCalculator::RenderToCpu(
        CalculatorContext *cc, const ImageFormat::Format &target_format,
        uchar *data_image, std::unique_ptr<cv::Mat> &image_mat)
    {
        auto output_frame = absl::make_unique<ImageFrame>(
            target_format, image_mat->cols, image_mat->rows);

        output_frame->CopyPixelData(target_format, image_mat->cols, image_mat->rows, data_image,
                                    ImageFrame::kDefaultAlignmentBoundary);

        if (cc->Outputs().HasTag(kImageFrameTag))
        {
            cc->Outputs()
                .Tag(kImageFrameTag)
                .Add(output_frame.release(), cc->InputTimestamp());
        }

        return absl::OkStatus();
    }

    absl::Status WarpAffineCalculator::CreateRenderTargetCpu(
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
            case ImageFormat::SBGR:
                *target_format = ImageFormat::SBGR;
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

    absl::Status WarpAffineCalculator::AffineTransform(CalculatorContext *cc, std::unique_ptr<cv::Mat> &image_mat, Tensor<double> _src, Tensor<double> _dst)
    {
        Mat mat_image_ = *image_mat.get();
        Mat clone_image = mat_image_.clone();

        Mat outImage = Mat(mat_image_.size(), mat_image_.type());
        Mat out = mat_image_.clone();

        for (int i = 0; i < 854; ++i)
        {
            if (i == 246)
            {
                int pointer = 0;
            }
            Tensor<double> __t1 = _src.index(vector<int>{i});
            Tensor<double> __t2 = _dst.index(vector<int>{i});

            vector<Point> t1;
            vector<Point> t2;

            for (int i = 0; i < 3; ++i)
            {
                t1.push_back(Point(
                    (int)(__t1.at(vector<int>{0, 3 * i})),
                    (int)(__t1.at(vector<int>{0, 3 * i + 1}))));
                t2.push_back(Point(
                    (int)(__t2.at(vector<int>{0, 3 * i})),
                    (int)(__t2.at(vector<int>{0, 3 * i + 1}))));
            }

            cv::Rect r1 = cv::boundingRect(t1);
            cv::Rect r2 = cv::boundingRect(t2);
            cv::Point2f srcTri[3];
            cv::Point2f dstTri[3];
            std::vector<cv::Point> t1Rect;
            std::vector<cv::Point> t2Rect;

            for (int i = 0; i < 3; ++i)
            {
                srcTri[i] = Point2f(t1[i].x - r1.x, t1[i].y - r1.y);
                dstTri[i] = Point2f(t2[i].x - r2.x, t2[i].y - r2.y);
                t1Rect.push_back(Point(t1[i].x - r1.x, t1[i].y - r1.y));
                t2Rect.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y));
            }

            Mat _dst;
            Mat mask = Mat::zeros(r2.height, r2.width, CV_8U);
            cv::fillConvexPoly(mask, t2Rect, Scalar(1.0, 1.0, 1.0), 16, 0);

            if (r1.x + r1.width < clone_image.cols && r1.x >= 0 && r1.x + r1.width >= 0 && r1.y >= 0 && r1.y < clone_image.rows && r1.y + r1.height < clone_image.rows)
            {
                Mat imgRect = mat_image_(Range(r1.y, r1.y + r1.height), Range(r1.x, r1.x + r1.width));
                Mat warpMat = getAffineTransform(srcTri, dstTri);
                warpAffine(imgRect, _dst, warpMat, mask.size());

                for (int i = r2.y; i < r2.y + r2.height; ++i)
                {
                    for (int j = r2.x; j < r2.x + r2.width; ++j)
                    {
                        if ((int)mask.at<uchar>(i - r2.y, j - r2.x) > 0)
                        {
                            out.at<Vec3b>(i, j) = _dst.at<Vec3b>(i - r2.y, j - r2.x);
                        }
                    }
                }
            }
        }
        out.copyTo(*image_mat);

        return absl::OkStatus();
    }
    REGISTER_CALCULATOR(WarpAffineCalculator);
} // namespace mediapipe