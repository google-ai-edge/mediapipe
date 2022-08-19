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
#include "mediapipe/framework/formats/landmark.pb.h"
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
    namespace
    {
        template <class LandmarkType>
        bool IsLandmarkVisibleAndPresent(const LandmarkType &landmark,
                                         bool utilize_visibility,
                                         float visibility_threshold,
                                         bool utilize_presence,
                                         float presence_threshold)
        {
            if (utilize_visibility && landmark.has_visibility() &&
                landmark.visibility() < visibility_threshold)
            {
                return false;
            }
            if (utilize_presence && landmark.has_presence() &&
                landmark.presence() < presence_threshold)
            {
                return false;
            }
            return true;
        }

        bool NormalizedtoPixelCoordinates(double normalized_x, double normalized_y, double normalized_z,
                                          int image_width, int image_height, double *x_px,
                                          double *y_px, double *z_px)
        {
            CHECK(x_px != nullptr);
            CHECK(y_px != nullptr);
            CHECK_GT(image_width, 0);
            CHECK_GT(image_height, 0);

            if (normalized_x < 0 || normalized_x > 1.0 || normalized_y < 0 ||
                normalized_y > 1.0 || normalized_z < 0 ||
                normalized_z > 1.0)
            {
                VLOG(1) << "Normalized coordinates must be between 0.0 and 1.0";
            }

            *x_px = static_cast<double>(normalized_x) * image_width;
            *y_px = static_cast<double>(normalized_y) * image_height;
            *z_px = static_cast<double>(normalized_z) * image_width;

            return true;
        }
    }
    namespace api2
    {
        class LandmarksToPointArrayCalculator : public Node
        {
        public:
            static constexpr Input<NormalizedLandmarkList> kNormLandmarks{"NORM_LANDMARKS"};
            static constexpr Input<pair<int, int>> kImageSize{"IMAGE_SIZE"};
            static constexpr Output<vector<Point3d>> kOut{"POINTS"};

            MEDIAPIPE_NODE_CONTRACT(kNormLandmarks, kImageSize, kOut);

            static absl::Status UpdateContract(CalculatorContract *cc)
            {
                RET_CHECK(kOut(cc).IsConnected())
                    << "At least one output stream is expected.";
                return absl::OkStatus();
            }

            absl::Status Process(CalculatorContext *cc) override
            {
                if (kNormLandmarks(cc).IsEmpty())
                {
                    return absl::OkStatus();
                }
                if (kImageSize(cc).IsEmpty())
                {
                    return absl::OkStatus();
                }

                const auto &size = *kImageSize(cc);

                MP_RETURN_IF_ERROR(GetPoints(cc, size));

                auto output_frame = absl::make_unique<vector<Point3d>>(point_array);

                kOut(cc).Send(std::move(output_frame));
                return absl::OkStatus();
            }

            absl::Status GetPoints(CalculatorContext *cc, pair<int, int> size)
            {

                const auto &landmarks = *kNormLandmarks(cc);
                point_array = {};
                for (int i = 0; i < landmarks.landmark_size(); i++)
                {
                    const NormalizedLandmark &landmark = landmarks.landmark(i);

                    if (!IsLandmarkVisibleAndPresent<NormalizedLandmark>(
                            landmark, false,
                            0.0, false,
                            0.0))
                    {
                        continue;
                    }

                    const auto &point = landmark;

                    double x = -1;
                    double y = -1;
                    double z = -1;

                    CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), point.z(), size.first,
                                                       size.second, &x, &y, &z));

                    point_array.push_back({x, y, z});
                }

                return absl::OkStatus();
            }

        private:
            std::vector<cv::Point3d> point_array;
        };
        MEDIAPIPE_REGISTER_NODE(LandmarksToPointArrayCalculator);
    }
} // namespace mediapipe