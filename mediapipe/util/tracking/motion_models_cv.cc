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

#include "mediapipe/util/tracking/motion_models_cv.h"

#include "absl/log/absl_check.h"

namespace mediapipe {

void ModelCvConvert<TranslationModel>::ToCvMat(const TranslationModel& model,
                                               cv::Mat* matrix) {
  ModelCvConvert<AffineModel>::ToCvMat(
      ModelAdapter<TranslationModel>::ToAffine(model), matrix);
}

void ModelCvConvert<LinearSimilarityModel>::ToCvMat(
    const LinearSimilarityModel& model, cv::Mat* matrix) {
  ModelCvConvert<AffineModel>::ToCvMat(
      ModelAdapter<LinearSimilarityModel>::ToAffine(model), matrix);
}

void ModelCvConvert<AffineModel>::ToCvMat(const AffineModel& model,
                                          cv::Mat* matrix) {
  matrix->create(2, 3, CV_32FC1);
  matrix->at<float>(0, 0) = model.a();
  matrix->at<float>(0, 1) = model.b();
  matrix->at<float>(0, 2) = model.dx();
  matrix->at<float>(1, 0) = model.c();
  matrix->at<float>(1, 1) = model.d();
  matrix->at<float>(1, 2) = model.dy();
}

void ModelCvConvert<Homography>::ToCvMat(const Homography& model,
                                         cv::Mat* matrix) {
  ABSL_CHECK(matrix != nullptr);
  matrix->create(3, 3, CV_32FC1);
  matrix->at<float>(0, 0) = model.h_00();
  matrix->at<float>(0, 1) = model.h_01();
  matrix->at<float>(0, 2) = model.h_02();
  matrix->at<float>(1, 0) = model.h_10();
  matrix->at<float>(1, 1) = model.h_11();
  matrix->at<float>(1, 2) = model.h_12();
  matrix->at<float>(2, 0) = model.h_20();
  matrix->at<float>(2, 1) = model.h_21();
  matrix->at<float>(2, 2) = 1.0;
}

}  // namespace mediapipe
