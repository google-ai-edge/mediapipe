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

#include "mediapipe/tasks/c/components/containers/landmark_converter.h"

#include <cstring>

#include "mediapipe/tasks/c/components/containers/landmark.h"

typedef Landmark LandmarkC;
typedef NormalizedLandmark NormalizedLandmarkC;
typedef Landmarks LandmarksC;
typedef NormalizedLandmarks NormalizedLandmarksC;

#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToLandmark(
    const mediapipe::tasks::components::containers::Landmark& in,
    LandmarkC* out) {
  out->x = in.x;
  out->y = in.y;
  out->z = in.z;

  if (in.visibility.has_value()) {
    out->has_visibility = true;
    out->visibility = in.visibility.value();
  } else {
    out->has_visibility = false;
  }

  if (in.presence.has_value()) {
    out->has_presence = true;
    out->presence = in.presence.value();
  } else {
    out->has_presence = false;
  }

  out->name = in.name.has_value() ? strdup(in.name->c_str()) : nullptr;
}

void CppConvertToNormalizedLandmark(
    const mediapipe::tasks::components::containers::NormalizedLandmark& in,
    NormalizedLandmarkC* out) {
  out->x = in.x;
  out->y = in.y;
  out->z = in.z;

  if (in.visibility.has_value()) {
    out->has_visibility = true;
    out->visibility = in.visibility.value();
  } else {
    out->has_visibility = false;
  }

  if (in.presence.has_value()) {
    out->has_presence = true;
    out->presence = in.presence.value();
  } else {
    out->has_presence = false;
  }

  out->name = in.name.has_value() ? strdup(in.name->c_str()) : nullptr;
}

void CppConvertToLandmarks(
    const std::vector<mediapipe::tasks::components::containers::Landmark>& in,
    LandmarksC* out) {
  out->landmarks_count = in.size();
  out->landmarks = new LandmarkC[out->landmarks_count];
  for (uint32_t i = 0; i < out->landmarks_count; ++i) {
    CppConvertToLandmark(in[i], &out->landmarks[i]);
  }
}

void CppConvertToNormalizedLandmarks(
    const std::vector<
        mediapipe::tasks::components::containers::NormalizedLandmark>& in,
    NormalizedLandmarksC* out) {
  out->landmarks_count = in.size();
  out->landmarks = new NormalizedLandmarkC[out->landmarks_count];
  for (uint32_t i = 0; i < out->landmarks_count; ++i) {
    CppConvertToNormalizedLandmark(in[i], &out->landmarks[i]);
  }
}

void CppCloseLandmark(LandmarkC* in) {
  if (in && in->name) {
    free(in->name);
    in->name = nullptr;
  }
}

void CppCloseLandmarks(LandmarksC* in) {
  for (uint32_t i = 0; i < in->landmarks_count; ++i) {
    CppCloseLandmark(&in->landmarks[i]);
  }
  delete[] in->landmarks;
  in->landmarks = nullptr;
  in->landmarks_count = 0;
}

void CppCloseNormalizedLandmark(NormalizedLandmarkC* in) {
  if (in && in->name) {
    free(in->name);
    in->name = nullptr;
  }
}

void CppCloseNormalizedLandmarks(NormalizedLandmarksC* in) {
  for (uint32_t i = 0; i < in->landmarks_count; ++i) {
    CppCloseNormalizedLandmark(&in->landmarks[i]);
  }
  delete[] in->landmarks;
  in->landmarks = nullptr;
  in->landmarks_count = 0;
}

}  // namespace mediapipe::tasks::c::components::containers
