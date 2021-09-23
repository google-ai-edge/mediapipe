// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/calculators/image/affine_transformation_runner_opencv.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/image/affine_transformation.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

cv::BorderTypes GetBorderModeForOpenCv(
    AffineTransformation::BorderMode border_mode) {
  switch (border_mode) {
    case AffineTransformation::BorderMode::kZero:
      return cv::BORDER_CONSTANT;
    case AffineTransformation::BorderMode::kReplicate:
      return cv::BORDER_REPLICATE;
  }
}

class OpenCvRunner
    : public AffineTransformation::Runner<ImageFrame, ImageFrame> {
 public:
  absl::StatusOr<ImageFrame> Run(
      const ImageFrame& input, const std::array<float, 16>& matrix,
      const AffineTransformation::Size& size,
      AffineTransformation::BorderMode border_mode) override {
    // OpenCV warpAffine works in absolute coordinates, so the transfom (which
    // accepts and produces relative coordinates) should be adjusted to first
    // normalize coordinates and then scale them.
    // clang-format off
    cv::Matx44f normalize_dst_coordinate({
      1.0f / size.width, 0.0f,               0.0f, 0.0f,
      0.0f,              1.0f / size.height, 0.0f, 0.0f,
      0.0f,              0.0f,               1.0f, 0.0f,
      0.0f,              0.0f,               0.0f, 1.0f});
    cv::Matx44f scale_src_coordinate({
      1.0f * input.Width(), 0.0f,                  0.0f, 0.0f,
      0.0f,                 1.0f * input.Height(), 0.0f, 0.0f,
      0.0f,                 0.0f,                  1.0f, 0.0f,
      0.0f,                 0.0f,                  0.0f, 1.0f});
    // clang-format on
    cv::Matx44f adjust_dst_coordinate;
    cv::Matx44f adjust_src_coordinate;
    // TODO: update to always use accurate implementation.
    constexpr bool kOpenCvCompatibility = true;
    if (kOpenCvCompatibility) {
      adjust_dst_coordinate = normalize_dst_coordinate;
      adjust_src_coordinate = scale_src_coordinate;
    } else {
      // To do an accurate affine image transformation and make "on-cpu" and
      // "on-gpu" calculations aligned - extra offset is required to select
      // correct pixels.
      //
      // Each destination pixel corresponds to some pixels region from source
      // image.(In case of downscaling there can be more than one pixel.) The
      // offset for x and y is calculated in the way, so pixel in the middle of
      // the region is selected.
      //
      // For simplicity sake, let's consider downscaling from 100x50 to 10x10
      // without a rotation:
      // 1. Each destination pixel corresponds to 10x5 region
      //    X range: [0, .. , 9]
      //    Y range: [0, .. , 4]
      // 2. Considering we have __discrete__ pixels, the center of the region is
      //    between (4, 2) and (5, 2) pixels, let's assume it's a "pixel"
      //    (4.5, 2).
      // 3. When using the above as an offset for every pixel select while
      //    downscaling, resulting pixels are:
      //      (4.5, 2), (14.5, 2), .. , (94.5, 2)
      //      (4.5, 7), (14.5, 7), .. , (94.5, 7)
      //      ..
      //      (4.5, 47), (14.5, 47), .., (94.5, 47)
      //    instead of:
      //      (0, 0), (10, 0), .. , (90, 0)
      //      (0, 5), (10, 7), .. , (90, 5)
      //      ..
      //      (0, 45), (10, 45), .., (90, 45)
      //    The latter looks shifted.
      //
      // Offsets are needed, so that __discrete__ pixel at (0, 0) corresponds to
      // the same pixel as would __non discrete__ pixel at (0.5, 0.5). Hence,
      // transformation matrix should shift coordinates by (0.5, 0.5) as the
      // very first step.
      //
      // Due to the above shift, transformed coordinates would be valid for
      // float coordinates where pixel (0, 0) spans [0.0, 1.0) x [0.0, 1.0).
      // T0 make it valid for __discrete__ pixels, transformation matrix should
      // shift coordinate by (-0.5f, -0.5f) as the very last step. (E.g. if we
      // get (0.5f, 0.5f), then it's (0, 0) __discrete__ pixel.)
      // clang-format off
      cv::Matx44f shift_dst({1.0f, 0.0f, 0.0f, 0.5f,
                              0.0f, 1.0f, 0.0f, 0.5f,
                              0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 1.0f});
      cv::Matx44f shift_src({1.0f, 0.0f, 0.0f, -0.5f,
                              0.0f, 1.0f, 0.0f, -0.5f,
                              0.0f, 0.0f, 1.0f,  0.0f,
                              0.0f, 0.0f, 0.0f,  1.0f});
      // clang-format on
      adjust_dst_coordinate = normalize_dst_coordinate * shift_dst;
      adjust_src_coordinate = shift_src * scale_src_coordinate;
    }

    cv::Matx44f transform(matrix.data());
    cv::Matx44f transform_absolute =
        adjust_src_coordinate * transform * adjust_dst_coordinate;

    cv::Mat in_mat = formats::MatView(&input);

    cv::Mat cv_affine_transform(2, 3, CV_32F);
    cv_affine_transform.at<float>(0, 0) = transform_absolute.val[0];
    cv_affine_transform.at<float>(0, 1) = transform_absolute.val[1];
    cv_affine_transform.at<float>(0, 2) = transform_absolute.val[3];
    cv_affine_transform.at<float>(1, 0) = transform_absolute.val[4];
    cv_affine_transform.at<float>(1, 1) = transform_absolute.val[5];
    cv_affine_transform.at<float>(1, 2) = transform_absolute.val[7];

    ImageFrame out_image(input.Format(), size.width, size.height);
    cv::Mat out_mat = formats::MatView(&out_image);

    cv::warpAffine(in_mat, out_mat, cv_affine_transform,
                   cv::Size(out_mat.cols, out_mat.rows),
                   /*flags=*/cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   GetBorderModeForOpenCv(border_mode));

    return out_image;
  }
};

}  // namespace

absl::StatusOr<
    std::unique_ptr<AffineTransformation::Runner<ImageFrame, ImageFrame>>>
CreateAffineTransformationOpenCvRunner() {
  return absl::make_unique<OpenCvRunner>();
}

}  // namespace mediapipe
