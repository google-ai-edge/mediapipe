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

#include "mediapipe/calculators/landmarks/point_vector_to_mask_calculator.h"

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
        class PointVectorToMaskCalculator : public Node
        {
        public:
            static constexpr Input<Mat> kImage{"IMAGE"};
            static constexpr Input<unordered_map<string, Mat>> kMasks{"MASKS"};
            static constexpr Input<tuple<double, double, double, double>> kFaceBox{"FACE_BOX"};
            static constexpr Output<Mat> kOut{"FOREHEAD_MASK"};

            MEDIAPIPE_NODE_CONTRACT(kImage, kMasks, kFaceBox, kOut);

            static absl::Status UpdateContract(CalculatorContract *cc)
            {
                RET_CHECK(kOut(cc).IsConnected())
                    << "At least one output stream is expected.";
                return absl::OkStatus();
            }

            absl::Status Process(CalculatorContext *cc) override
            {
                if (kImage(cc).IsEmpty() || kMasks(cc).IsEmpty() || kFaceBox(cc).IsEmpty())
                {
                    return absl::OkStatus();
                }

                const auto &mat_image_ = *kImage(cc);
                const auto &masks = *kMasks(cc);
                const auto &face_box = *kFaceBox(cc);

                MP_RETURN_IF_ERROR(PredictForeheadMask(cc, mat_image_, masks, face_box));

                auto output_frame = absl::make_unique<Mat>(new_skin_mask);

                kOut(cc).Send(std::move(output_frame));

                return absl::OkStatus();
            }

            absl::Status PredictForeheadMask(CalculatorContext *cc, Mat mat_image_, 
            unordered_map<string, Mat> mask_vec, tuple<double, double, double, double> face_box)
            {

                cv::Mat part_forehead_mask = mask_vec.find("PART_FOREHEAD_B")->second.clone();
                part_forehead_mask.convertTo(part_forehead_mask, CV_32F, 1.0 / 255);
                part_forehead_mask.convertTo(part_forehead_mask, CV_8U);

                cv::Mat image_sm, image_sm_hsv, skinMask;

                cv::resize(mat_image_, image_sm, cv::Size(mat_image_.cols, mat_image_.rows));
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

                new_skin_mask(cv::Range(get<1>(face_box), max_part_f), cv::Range(x_min_part, x_max_part)) =
                    skinMask(cv::Range(get<1>(face_box), max_part_f), cv::Range(x_min_part, x_max_part));

                return absl::OkStatus();
            }

        private:
            Mat new_skin_mask;
        };
        MEDIAPIPE_REGISTER_NODE(PointVectorToMaskCalculator);
    }
} // namespace mediapipe