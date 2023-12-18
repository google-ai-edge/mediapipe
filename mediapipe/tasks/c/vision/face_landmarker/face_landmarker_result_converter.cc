/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result_converter.h"

#include <cstdint>
#include <vector>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/landmark_converter.h"
#include "mediapipe/tasks/c/components/containers/matrix.h"
#include "mediapipe/tasks/c/components/containers/matrix_converter.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

using CppCategory = ::mediapipe::tasks::components::containers::Category;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;

void CppConvertToFaceLandmarkerResult(
    const ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult& in,
    FaceLandmarkerResult* out) {
  out->face_landmarks_count = in.face_landmarks.size();
  out->face_landmarks = new NormalizedLandmarks[out->face_landmarks_count];
  for (uint32_t i = 0; i < out->face_landmarks_count; ++i) {
    std::vector<CppNormalizedLandmark> cpp_normalized_landmarks;
    for (uint32_t j = 0; j < in.face_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.face_landmarks[i].landmarks[j];
      cpp_normalized_landmarks.push_back(cpp_landmark);
    }
    CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                    &out->face_landmarks[i]);
  }

  if (in.face_blendshapes.has_value()) {
    out->face_blendshapes_count = in.face_blendshapes->size();
    out->face_blendshapes = new Categories[out->face_blendshapes_count];

    for (uint32_t i = 0; i < out->face_blendshapes_count; ++i) {
      uint32_t categories_count =
          in.face_blendshapes.value()[i].categories.size();
      out->face_blendshapes[i].categories_count = categories_count;
      out->face_blendshapes[i].categories = new Category[categories_count];

      for (uint32_t j = 0; j < categories_count; ++j) {
        const auto& cpp_category = in.face_blendshapes.value()[i].categories[j];
        CppConvertToCategory(cpp_category,
                             &out->face_blendshapes[i].categories[j]);
      }
    }
  } else {
    out->face_blendshapes_count = 0;
    out->face_blendshapes = nullptr;
  }

  if (in.facial_transformation_matrixes.has_value()) {
    out->facial_transformation_matrixes_count =
        in.facial_transformation_matrixes.value().size();
    out->facial_transformation_matrixes =
        new ::Matrix[out->facial_transformation_matrixes_count];
    for (uint32_t i = 0; i < out->facial_transformation_matrixes_count; ++i) {
      CppConvertToMatrix(in.facial_transformation_matrixes.value()[i],
                         &out->facial_transformation_matrixes[i]);
    }
  } else {
    out->facial_transformation_matrixes_count = 0;
    out->facial_transformation_matrixes = nullptr;
  }
}

void CppCloseFaceLandmarkerResult(FaceLandmarkerResult* result) {
  for (uint32_t i = 0; i < result->face_blendshapes_count; ++i) {
    for (uint32_t j = 0; j < result->face_blendshapes[i].categories_count;
         ++j) {
      CppCloseCategory(&result->face_blendshapes[i].categories[j]);
    }
    delete[] result->face_blendshapes[i].categories;
  }
  delete[] result->face_blendshapes;

  for (uint32_t i = 0; i < result->face_landmarks_count; ++i) {
    CppCloseNormalizedLandmarks(&result->face_landmarks[i]);
  }
  delete[] result->face_landmarks;

  for (uint32_t i = 0; i < result->facial_transformation_matrixes_count; ++i) {
    CppCloseMatrix(&result->facial_transformation_matrixes[i]);
  }
  delete[] result->facial_transformation_matrixes;

  result->face_blendshapes_count = 0;
  result->face_landmarks_count = 0;
  result->facial_transformation_matrixes_count = 0;
  result->face_blendshapes = nullptr;
  result->face_landmarks = nullptr;
  result->facial_transformation_matrixes = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
