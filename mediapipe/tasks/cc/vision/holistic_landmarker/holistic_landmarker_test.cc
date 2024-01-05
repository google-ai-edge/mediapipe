/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/test_util.h"
#include "util/tuple/dump_vars.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::file::Defaults;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::ConvertToLandmarks;
using ::mediapipe::tasks::components::containers::ConvertToNormalizedLandmarks;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kHolisticLandmarkerBundleAsset[] = "holistic_landmarker.task";
constexpr char kHolisticLandmarksFilename[] = "male_full_height_hands_result_cpu.pbtxt";

constexpr char kPoseImage[] = "male_full_height_hands.jpg";
constexpr char kCatImage[] = "cat.jpg";

constexpr float kLandmarksAbsMargin = 0.03;
constexpr float kLandmarksOnVideoAbsMargin = 0.03;


}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
