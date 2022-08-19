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
using namespace cv;

namespace mediapipe
{
    namespace api2
    {
        class PointsToFaceBoxCalculator : public Node
        {
        public:
            static constexpr Input<vector<Point3d>> kPoints{"POINTS"};
            static constexpr Input<pair<int, int>> kImageSize{"IMAGE_SIZE"};
            static constexpr Output<tuple<double, double, double, double>> kOut{"FACE_BOX"};

            MEDIAPIPE_NODE_CONTRACT(kPoints, kImageSize, kOut);

            static absl::Status UpdateContract(CalculatorContract *cc)
            {
                RET_CHECK(kOut(cc).IsConnected())
                    << "At least one output stream is expected.";
                return absl::OkStatus();
            }

            absl::Status Process(CalculatorContext *cc) override
            {
                if (kPoints(cc).IsEmpty())
                {
                    return absl::OkStatus();
                }
                if (kImageSize(cc).IsEmpty())
                {
                    return absl::OkStatus();
                }

                const auto &size = *kImageSize(cc);
                const auto &points = *kPoints(cc);

                MP_RETURN_IF_ERROR(GetFaceBox(cc, size, points));

                auto output_frame = absl::make_unique<tuple<double, double, double, double>>(face_box);

                kOut(cc).Send(std::move(output_frame));

                return absl::OkStatus();
            }

            absl::Status GetFaceBox(CalculatorContext *cc, pair<int, int> size, vector<Point3d> points)
            {
                cv::Mat points_mat(points);

                points_mat.convertTo(points_mat, CV_32F);
                cv::Mat min, max;

                cv::reduce(points_mat, min, 0, CV_REDUCE_MIN, CV_32F);
                cv::reduce(points_mat, max, 0, CV_REDUCE_MAX, CV_32F);

                min.at<float>(0, 1) *= 0.9;
                face_box = {min.at<float>(0, 0), min.at<float>(0, 1), max.at<float>(0, 0), max.at<float>(0, 1)};

                return absl::OkStatus();
            }

        private:
            tuple<double, double, double, double> face_box;
        };
        MEDIAPIPE_REGISTER_NODE(PointsToFaceBoxCalculator);
    }
} // namespace mediapipe