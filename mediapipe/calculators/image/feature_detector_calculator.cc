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

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/synchronization/blocking_counter.h"
#include "mediapipe/calculators/image/feature_detector_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_features2d_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/tool/options_util.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

const char kOptionsTag[] = "OPTIONS";
const int kPatchSize = 32;
const int kNumThreads = 16;

// A calculator to apply local feature detection.
// Input stream:
//   IMAGE: Input image frame of type ImageFrame from video stream.
// Output streams:
//   FEATURES: The detected keypoints from input image as vector<cv::KeyPoint>.
//   PATCHES:  Optional output the extracted patches as vector<cv::Mat>
class FeatureDetectorCalculator : public CalculatorBase {
 public:
  ~FeatureDetectorCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  FeatureDetectorCalculatorOptions options_;
  cv::Ptr<cv::Feature2D> feature_detector_;
  std::unique_ptr<::mediapipe::ThreadPool> pool_;

  // Create image pyramid based on input image.
  void ComputeImagePyramid(const cv::Mat& input_image,
                           std::vector<cv::Mat>* image_pyramid);

  // Extract the patch for single feature with image pyramid.
  cv::Mat ExtractPatch(const cv::KeyPoint& feature,
                       const std::vector<cv::Mat>& image_pyramid);
};

REGISTER_CALCULATOR(FeatureDetectorCalculator);

::mediapipe::Status FeatureDetectorCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->Inputs().HasTag("IMAGE")) {
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag("FEATURES")) {
    cc->Outputs().Tag("FEATURES").Set<std::vector<cv::KeyPoint>>();
  }
  if (cc->Outputs().HasTag("LANDMARKS")) {
    cc->Outputs().Tag("LANDMARKS").Set<NormalizedLandmarkList>();
  }
  if (cc->Outputs().HasTag("PATCHES")) {
    cc->Outputs().Tag("PATCHES").Set<std::vector<TfLiteTensor>>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status FeatureDetectorCalculator::Open(CalculatorContext* cc) {
  options_ =
      tool::RetrieveOptions(cc->Options(), cc->InputSidePackets(), kOptionsTag)
          .GetExtension(FeatureDetectorCalculatorOptions::ext);
  feature_detector_ = cv::ORB::create(
      options_.max_features(), options_.scale_factor(),
      options_.pyramid_level(), kPatchSize - 1, 0, 2, cv::ORB::FAST_SCORE);
  pool_ = absl::make_unique<::mediapipe::ThreadPool>("ThreadPool", kNumThreads);
  pool_->StartWorkers();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status FeatureDetectorCalculator::Process(CalculatorContext* cc) {
  const Timestamp& timestamp = cc->InputTimestamp();
  if (timestamp == Timestamp::PreStream()) {
    // Indicator packet.
    return ::mediapipe::OkStatus();
  }
  InputStream* input_frame = &(cc->Inputs().Tag("IMAGE"));
  cv::Mat input_view = formats::MatView(&input_frame->Get<ImageFrame>());
  cv::Mat grayscale_view;
  cv::cvtColor(input_view, grayscale_view, cv::COLOR_RGB2GRAY);

  std::vector<cv::KeyPoint> keypoints;
  feature_detector_->detect(grayscale_view, keypoints);
  if (keypoints.size() > options_.max_features()) {
    keypoints.resize(options_.max_features());
  }

  if (cc->Outputs().HasTag("FEATURES")) {
    auto features_ptr = absl::make_unique<std::vector<cv::KeyPoint>>(keypoints);
    cc->Outputs().Tag("FEATURES").Add(features_ptr.release(), timestamp);
  }

  if (cc->Outputs().HasTag("LANDMARKS")) {
    auto landmarks_ptr = absl::make_unique<NormalizedLandmarkList>();
    for (int j = 0; j < keypoints.size(); ++j) {
      auto feature_landmark = landmarks_ptr->add_landmark();
      feature_landmark->set_x(keypoints[j].pt.x / grayscale_view.cols);
      feature_landmark->set_y(keypoints[j].pt.y / grayscale_view.rows);
    }
    cc->Outputs().Tag("LANDMARKS").Add(landmarks_ptr.release(), timestamp);
  }

  if (cc->Outputs().HasTag("PATCHES")) {
    std::vector<cv::Mat> image_pyramid;
    ComputeImagePyramid(grayscale_view, &image_pyramid);
    std::vector<cv::Mat> patch_mat;
    patch_mat.resize(keypoints.size());
    absl::BlockingCounter counter(keypoints.size());
    for (int i = 0; i < keypoints.size(); i++) {
      pool_->Schedule(
          [this, &image_pyramid, &keypoints, &patch_mat, i, &counter] {
            patch_mat[i] = ExtractPatch(keypoints[i], image_pyramid);
            counter.DecrementCount();
          });
    }
    counter.Wait();
    const int batch_size = options_.max_features();
    auto patches = absl::make_unique<std::vector<TfLiteTensor>>();
    TfLiteTensor tensor;
    tensor.type = kTfLiteFloat32;
    tensor.dims = TfLiteIntArrayCreate(4);
    tensor.dims->data[0] = batch_size;
    tensor.dims->data[1] = kPatchSize;
    tensor.dims->data[2] = kPatchSize;
    tensor.dims->data[3] = 1;
    int num_bytes = batch_size * kPatchSize * kPatchSize * sizeof(float);
    tensor.data.data = malloc(num_bytes);
    tensor.bytes = num_bytes;
    tensor.allocation_type = kTfLiteArenaRw;
    float* tensor_buffer = tensor.data.f;
    for (int i = 0; i < keypoints.size(); i++) {
      for (int j = 0; j < patch_mat[i].rows; ++j) {
        for (int k = 0; k < patch_mat[i].cols; ++k) {
          *tensor_buffer++ = patch_mat[i].at<uchar>(j, k) / 128.0f - 1.0f;
        }
      }
    }
    for (int i = keypoints.size() * kPatchSize * kPatchSize; i < num_bytes / 4;
         i++) {
      *tensor_buffer++ = 0;
    }

    patches->emplace_back(tensor);
    cc->Outputs().Tag("PATCHES").Add(patches.release(), timestamp);
  }

  return ::mediapipe::OkStatus();
}

void FeatureDetectorCalculator::ComputeImagePyramid(
    const cv::Mat& input_image, std::vector<cv::Mat>* image_pyramid) {
  cv::Mat tmp_image = input_image;
  cv::Mat src_image = input_image;
  for (int i = 0; i < options_.pyramid_level(); ++i) {
    image_pyramid->push_back(src_image);
    cv::resize(src_image, tmp_image, cv::Size(), 1.0f / options_.scale_factor(),
               1.0f / options_.scale_factor());
    src_image = tmp_image;
  }
}

cv::Mat FeatureDetectorCalculator::ExtractPatch(
    const cv::KeyPoint& feature, const std::vector<cv::Mat>& image_pyramid) {
  cv::Mat img = image_pyramid[feature.octave];
  float scale_factor = 1 / pow(options_.scale_factor(), feature.octave);
  cv::Point2f center =
      cv::Point2f(feature.pt.x * scale_factor, feature.pt.y * scale_factor);
  cv::Mat rot = cv::getRotationMatrix2D(center, feature.angle, 1.0);
  rot.at<double>(0, 2) += kPatchSize / 2 - center.x;
  rot.at<double>(1, 2) += kPatchSize / 2 - center.y;
  cv::Mat cropped_img;
  // perform the affine transformation
  cv::warpAffine(img, cropped_img, rot, cv::Size(kPatchSize, kPatchSize),
                 cv::INTER_LINEAR);
  return cropped_img;
}

}  // namespace mediapipe
