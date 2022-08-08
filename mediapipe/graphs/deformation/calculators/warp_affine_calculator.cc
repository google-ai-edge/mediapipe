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
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe
{
    namespace api2
    {
        class WarpAffineCalculator : public Node
        {
        public:
            static constexpr Input<ImageFrame> kImageFrame{"IMAGE"};
            static constexpr Input<Tensor<double>>::Optional kSrc{"SRC_TENSOR"};
            static constexpr Input<Tensor<double>>::Optional kDst{"DST_TENSOR"};
            static constexpr Output<ImageFrame> kOut{"IMAGE"};

            MEDIAPIPE_NODE_CONTRACT(kImageFrame, kSrc, kDst, kOut);

            static absl::Status UpdateContract(CalculatorContract *cc)
            {
                RET_CHECK(kOut(cc).IsConnected())
                    << "At least one output stream is expected.";
                return absl::OkStatus();
            }

            absl::Status Process(CalculatorContext *cc) override
            {
                if (kImageFrame(cc).IsEmpty())
                {
                    return absl::OkStatus();
                }

                const ImageFrame &input = *kImageFrame(cc);
                auto image = absl::make_unique<cv::Mat>(
                    input.Height(), input.Width(), CV_8UC3);

                auto mat_image_ = formats::MatView(&input);
                Mat out = mat_image_.clone();

                if (!kSrc(cc).IsEmpty() && !kDst(cc).IsEmpty())
                {
                    Tensor<double> _src = (*kSrc(cc));
                    Tensor<double> _dst = (*kDst(cc));

                    Mat clone_image = mat_image_.clone();

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
                }

                out.copyTo(*image);

                auto output_frame = absl::make_unique<ImageFrame>(
                    input.Format(), mat_image_.cols, mat_image_.rows);
                uchar *image_mat_ptr = image->data;
                output_frame->CopyPixelData(input.Format(), mat_image_.cols, mat_image_.rows, image_mat_ptr,
                                            ImageFrame::kDefaultAlignmentBoundary);

                kOut(cc).Send(std::move(output_frame));
                return absl::OkStatus();
            }
        };

        MEDIAPIPE_REGISTER_NODE(WarpAffineCalculator);
    }
} // namespace mediapipe