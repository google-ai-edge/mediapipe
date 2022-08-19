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

using namespace std;
using namespace cv;

namespace mediapipe
{
    namespace api2
    {
        class PointVectorToMaskCalculator : public Node
        {
        public:
            static constexpr Input<vector<Point3d>> kPoints{"POINTS"};
            static constexpr Input<pair<int, int>> kImageSize{"IMAGE_SIZE"};
            static constexpr Output<std::unordered_map<std::string, cv::Mat>> kOut{"MASKS"};

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

                MP_RETURN_IF_ERROR(GetMasks(cc, size, points));

                auto output_frame = absl::make_unique<std::unordered_map<std::string, cv::Mat>>(all_masks);

                kOut(cc).Send(std::move(output_frame), cc->InputTimestamp());

                return absl::OkStatus();
            }

            absl::Status GetMasks(CalculatorContext *cc, pair<int, int> size, vector<Point3d> points)
            {
                all_masks= {};

                for (const auto &[key, value] : orderList)
                {
                    vector<vector<Point>> point_vec = {};
                    vector<Point> point_array = {};
                    for (auto order : value)
                    {
                        point_array.push_back({points[order].x, points[order].y});
                    }
                    point_vec.push_back(point_array);
                    
                    cv::Mat mask = cv::Mat::zeros({size.first, size.second}, CV_32FC1);
                    cv::fillPoly(mask, point_vec, cv::Scalar::all(255), cv::LINE_AA);
                    mask.convertTo(mask, CV_8U);

                    all_masks.insert({key, mask});
                }
                return absl::OkStatus();
            }

        private:
            std::unordered_map<std::string, cv::Mat> all_masks;
        };
        MEDIAPIPE_REGISTER_NODE(PointVectorToMaskCalculator);
    }
} // namespace mediapipe