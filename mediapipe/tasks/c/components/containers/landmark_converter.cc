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

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToLandmark(
    const mediapipe::tasks::components::containers::Landmark& in,
    ::Landmark* out) {
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
    ::NormalizedLandmark* out) {
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
    ::Landmarks* out) {
  out->landmarks_count = in.size();
  out->landmarks = new ::Landmark[out->landmarks_count];
  for (uint32_t i = 0; i < out->landmarks_count; ++i) {
    CppConvertToLandmark(in[i], &out->landmarks[i]);
  }
}

void CppConvertToNormalizedLandmarks(
    const std::vector<
        mediapipe::tasks::components::containers::NormalizedLandmark>& in,
    ::NormalizedLandmarks* out) {
  out->landmarks_count = in.size();
  out->landmarks = new ::NormalizedLandmark[out->landmarks_count];
  for (uint32_t i = 0; i < out->landmarks_count; ++i) {
    CppConvertToNormalizedLandmark(in[i], &out->landmarks[i]);
  }
}

void CppCloseLandmark(::Landmark* in) {
  if (in && in->name) {
    free(in->name);
    in->name = nullptr;
  }
}

void CppCloseLandmarks(::Landmarks* in) {
  for (uint32_t i = 0; i < in->landmarks_count; ++i) {
    CppCloseLandmark(&in->landmarks[i]);
  }
  delete[] in->landmarks;
  in->landmarks = nullptr;
  in->landmarks_count = 0;
}

void CppCloseNormalizedLandmark(::NormalizedLandmark* in) {
  if (in && in->name) {
    free(in->name);
    in->name = nullptr;
  }
}

void CppCloseNormalizedLandmarks(::NormalizedLandmarks* in) {
  for (uint32_t i = 0; i < in->landmarks_count; ++i) {
    CppCloseNormalizedLandmark(&in->landmarks[i]);
  }
  delete[] in->landmarks;
  in->landmarks = nullptr;
  in->landmarks_count = 0;
}

}  // namespace mediapipe::tasks::c::components::containers
