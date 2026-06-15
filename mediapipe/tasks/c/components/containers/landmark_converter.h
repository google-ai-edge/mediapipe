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

#ifndef MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_LANDMARK_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_LANDMARK_CONVERTER_H_

#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToLandmark(
    const mediapipe::tasks::components::containers::Landmark& in,
    MpLandmark* out);

void CppConvertToNormalizedLandmark(
    const mediapipe::tasks::components::containers::NormalizedLandmark& in,
    MpNormalizedLandmark* out);

void CppConvertToLandmarks(
    const std::vector<mediapipe::tasks::components::containers::Landmark>& in,
    MpLandmarks* out);

void CppConvertToNormalizedLandmarks(
    const std::vector<
        mediapipe::tasks::components::containers::NormalizedLandmark>& in,
    MpNormalizedLandmarks* out);

void CppCloseLandmark(struct MpLandmark* in);

void CppCloseLandmarks(struct MpLandmarks* in);

void CppCloseNormalizedLandmark(struct MpNormalizedLandmark* in);

void CppCloseNormalizedLandmarks(struct MpNormalizedLandmarks* in);

}  // namespace mediapipe::tasks::c::components::containers

#endif  // MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_LANDMARK_CONVERTER_H_
