// Copyright 2020 The MediaPipe Authors.
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

#include <cmath>
#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_file_properties.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.pb.h"

namespace mediapipe {

namespace {

constexpr char kIrisTag[] = "IRIS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kFocalLengthPixelTag[] = "FOCAL_LENGTH";
constexpr char kImageFilePropertiesTag[] = "IMAGE_FILE_PROPERTIES";
constexpr char kLeftIrisDepthTag[] = "LEFT_IRIS_DEPTH_MM";
constexpr char kRightIrisDepthTag[] = "RIGHT_IRIS_DEPTH_MM";
constexpr int kNumIrisLandmarksPerEye = 5;
constexpr float kDepthWeightUpdate = 0.1;
// Avergae fixed iris size across human beings.
constexpr float kIrisSizeInMM = 11.8;

inline float GetDepth(float x0, float y0, float x1, float y1) {
  return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

inline float GetLandmarkDepth(const NormalizedLandmark& ld0,
                              const NormalizedLandmark& ld1,
                              const std::pair<int, int>& image_size) {
  return GetDepth(ld0.x() * image_size.first, ld0.y() * image_size.second,
                  ld1.x() * image_size.first, ld1.y() * image_size.second);
}

float CalculateIrisDiameter(const NormalizedLandmarkList& landmarks,
                            const std::pair<int, int>& image_size) {
  const float dist_vert = GetLandmarkDepth(landmarks.landmark(1),
                                           landmarks.landmark(2), image_size);
  const float dist_hori = GetLandmarkDepth(landmarks.landmark(3),
                                           landmarks.landmark(4), image_size);
  return (dist_hori + dist_vert) / 2.0f;
}

float CalculateDepth(const NormalizedLandmark& center, float focal_length,
                     float iris_size, float img_w, float img_h) {
  std::pair<float, float> origin{img_w / 2.f, img_h / 2.f};
  const auto y = GetDepth(origin.first, origin.second, center.x() * img_w,
                          center.y() * img_h);
  const auto x = std::sqrt(focal_length * focal_length + y * y);
  const auto depth = kIrisSizeInMM * x / iris_size;
  return depth;
}

}  // namespace

// Estimates depth from iris to camera given focal length and image size.
//
// Usage example:
// node {
//   calculator: "IrisToDepthCalculator"
//   # A NormalizedLandmarkList contains landmarks for both iris.
//   input_stream: "IRIS:iris_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   # Note: Only one of FOCAL_LENGTH or IMAGE_FILE_PROPERTIES is necessary
//   # to get focal length in pixels. Sending focal length in pixels to
//   # this calculator is optional.
//   input_side_packet: "FOCAL_LENGTH:focal_length_pixel"
//   # OR
//   input_side_packet: "IMAGE_FILE_PROPERTIES:image_file_properties"
//   output_stream: "LEFT_IRIS_DEPTH_MM:left_iris_depth_mm"
//   output_stream: "RIGHT_IRIS_DEPTH_MM:right_iris_depth_mm"
// }
class IrisToDepthCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kIrisTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();

    // Only one of kFocalLengthPixelTag or kImageFilePropertiesTag must exist
    // if they are present.
    RET_CHECK(!(cc->InputSidePackets().HasTag(kFocalLengthPixelTag) &&
                cc->InputSidePackets().HasTag(kImageFilePropertiesTag)));
    if (cc->InputSidePackets().HasTag(kFocalLengthPixelTag)) {
      cc->InputSidePackets().Tag(kFocalLengthPixelTag).SetAny();
    }
    if (cc->InputSidePackets().HasTag(kImageFilePropertiesTag)) {
      cc->InputSidePackets()
          .Tag(kImageFilePropertiesTag)
          .Set<ImageFileProperties>();
    }
    if (cc->Outputs().HasTag(kLeftIrisDepthTag)) {
      cc->Outputs().Tag(kLeftIrisDepthTag).Set<float>();
    }
    if (cc->Outputs().HasTag(kRightIrisDepthTag)) {
      cc->Outputs().Tag(kRightIrisDepthTag).Set<float>();
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  float focal_length_pixels_ = -1.f;
  // TODO: Consolidate the logic when switching to input stream for
  // focal length.
  bool compute_depth_from_iris_ = false;
  float smoothed_left_depth_mm_ = -1.f;
  float smoothed_right_depth_mm_ = -1.f;

  void GetLeftIris(const NormalizedLandmarkList& lds,
                   NormalizedLandmarkList* iris);
  void GetRightIris(const NormalizedLandmarkList& lds,
                    NormalizedLandmarkList* iris);
  ::mediapipe::IrisToDepthCalculatorOptions options_;
};
REGISTER_CALCULATOR(IrisToDepthCalculator);

::mediapipe::Status IrisToDepthCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  if (cc->InputSidePackets().HasTag(kFocalLengthPixelTag)) {
#if defined(__APPLE__)
    focal_length_pixels_ = *cc->InputSidePackets()
                                .Tag(kFocalLengthPixelTag)
                                .Get<std::unique_ptr<float>>();
#else
    focal_length_pixels_ =
        cc->InputSidePackets().Tag(kFocalLengthPixelTag).Get<float>();
#endif
    compute_depth_from_iris_ = true;
  } else if (cc->InputSidePackets().HasTag(kImageFilePropertiesTag)) {
    const auto& properties = cc->InputSidePackets()
                                 .Tag(kImageFilePropertiesTag)
                                 .Get<ImageFileProperties>();
    focal_length_pixels_ = properties.focal_length_pixels();
    compute_depth_from_iris_ = true;
  }

  options_ = cc->Options<::mediapipe::IrisToDepthCalculatorOptions>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status IrisToDepthCalculator::Process(CalculatorContext* cc) {
  // Only process if there's input landmarks.
  if (cc->Inputs().Tag(kIrisTag).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }

  const auto& iris_landmarks =
      cc->Inputs().Tag(kIrisTag).Get<NormalizedLandmarkList>();
  RET_CHECK_EQ(iris_landmarks.landmark_size(), kNumIrisLandmarksPerEye * 2)
      << "Wrong number of iris landmarks";

  std::pair<int, int> image_size;
  RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());
  image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

  auto left_iris = absl::make_unique<NormalizedLandmarkList>();
  auto right_iris = absl::make_unique<NormalizedLandmarkList>();
  GetLeftIris(iris_landmarks, left_iris.get());
  GetRightIris(iris_landmarks, right_iris.get());

  const auto left_iris_size = CalculateIrisDiameter(*left_iris, image_size);
  const auto right_iris_size = CalculateIrisDiameter(*right_iris, image_size);

#if defined(__APPLE__)
  if (cc->InputSidePackets().HasTag(kFocalLengthPixelTag)) {
    focal_length_pixels_ = *cc->InputSidePackets()
                                .Tag(kFocalLengthPixelTag)
                                .Get<std::unique_ptr<float>>();
  }
#endif

  if (compute_depth_from_iris_ && focal_length_pixels_ > 0) {
    const auto left_depth =
        CalculateDepth(left_iris->landmark(0), focal_length_pixels_,
                       left_iris_size, image_size.first, image_size.second);
    const auto right_depth =
        CalculateDepth(right_iris->landmark(0), focal_length_pixels_,
                       right_iris_size, image_size.first, image_size.second);
    smoothed_left_depth_mm_ =
        smoothed_left_depth_mm_ < 0 || std::isinf(smoothed_left_depth_mm_)
            ? left_depth
            : smoothed_left_depth_mm_ * (1 - kDepthWeightUpdate) +
                  left_depth * kDepthWeightUpdate;
    smoothed_right_depth_mm_ =
        smoothed_right_depth_mm_ < 0 || std::isinf(smoothed_right_depth_mm_)
            ? right_depth
            : smoothed_right_depth_mm_ * (1 - kDepthWeightUpdate) +
                  right_depth * kDepthWeightUpdate;

    if (cc->Outputs().HasTag(kLeftIrisDepthTag)) {
      cc->Outputs()
          .Tag(kLeftIrisDepthTag)
          .AddPacket(MakePacket<float>(smoothed_left_depth_mm_)
                         .At(cc->InputTimestamp()));
    }
    if (cc->Outputs().HasTag(kRightIrisDepthTag)) {
      cc->Outputs()
          .Tag(kRightIrisDepthTag)
          .AddPacket(MakePacket<float>(smoothed_right_depth_mm_)
                         .At(cc->InputTimestamp()));
    }
  }
  return ::mediapipe::OkStatus();
}

void IrisToDepthCalculator::GetLeftIris(const NormalizedLandmarkList& lds,
                                        NormalizedLandmarkList* iris) {
  // Center, top, bottom, left, right
  *iris->add_landmark() = lds.landmark(options_.left_iris_center_index());
  *iris->add_landmark() = lds.landmark(options_.left_iris_top_index());
  *iris->add_landmark() = lds.landmark(options_.left_iris_bottom_index());
  *iris->add_landmark() = lds.landmark(options_.left_iris_left_index());
  *iris->add_landmark() = lds.landmark(options_.left_iris_right_index());
}

void IrisToDepthCalculator::GetRightIris(const NormalizedLandmarkList& lds,
                                         NormalizedLandmarkList* iris) {
  // Center, top, bottom, left, right
  *iris->add_landmark() = lds.landmark(options_.right_iris_center_index());
  *iris->add_landmark() = lds.landmark(options_.right_iris_top_index());
  *iris->add_landmark() = lds.landmark(options_.right_iris_bottom_index());
  *iris->add_landmark() = lds.landmark(options_.right_iris_left_index());
  *iris->add_landmark() = lds.landmark(options_.right_iris_right_index());
}
}  // namespace mediapipe
