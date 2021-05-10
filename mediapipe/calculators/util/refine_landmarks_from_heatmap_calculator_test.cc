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

#include "mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

mediapipe::NormalizedLandmarkList vec_to_lms(
    const std::vector<std::pair<float, float>>& inp) {
  mediapipe::NormalizedLandmarkList ret;
  for (const auto& it : inp) {
    auto new_lm = ret.add_landmark();
    new_lm->set_x(it.first);
    new_lm->set_y(it.second);
  }
  return ret;
}

std::vector<std::pair<float, float>> lms_to_vec(
    const mediapipe::NormalizedLandmarkList& lst) {
  std::vector<std::pair<float, float>> ret;
  for (const auto& lm : lst.landmark()) {
    ret.push_back({lm.x(), lm.y()});
  }
  return ret;
}

std::vector<float> CHW_to_HWC(std::vector<float> inp, int height, int width,
                              int depth) {
  std::vector<float> ret(inp.size());
  const float* inp_ptr = inp.data();
  for (int c = 0; c < depth; ++c) {
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int dest_idx = width * depth * row + depth * col + c;
        ret[dest_idx] = *inp_ptr;
        ++inp_ptr;
      }
    }
  }
  return ret;
}

using testing::ElementsAre;
using testing::FloatEq;
using testing::Pair;

TEST(RefineLandmarksFromHeatmapTest, Smoke) {
  float z = -10000000000000000;
  // clang-format off
  std::vector<float> hm = {
    z, z, z,
    1, z, z,
    z, z, z};
  // clang-format on

  auto ret_or_error = RefineLandmarksFromHeatMap(
      vec_to_lms({{0.5, 0.5}}), hm.data(), {3, 3, 1}, 3, 0.1, true, true);
  MP_EXPECT_OK(ret_or_error);
  EXPECT_THAT(lms_to_vec(*ret_or_error),
              ElementsAre(Pair(FloatEq(0), FloatEq(1 / 3.))));
}

TEST(RefineLandmarksFromHeatmapTest, MultiLayer) {
  float z = -10000000000000000;
  // clang-format off
  std::vector<float> hm = CHW_to_HWC({
    z, z, z,
    1, z, z,
    z, z, z,
    z, z, z,
    1, z, z,
    z, z, z,
    z, z, z,
    1, z, z,
    z, z, z}, 3, 3, 3);
  // clang-format on

  auto ret_or_error = RefineLandmarksFromHeatMap(
      vec_to_lms({{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}), hm.data(), {3, 3, 3}, 3,
      0.1, true, true);
  MP_EXPECT_OK(ret_or_error);
  EXPECT_THAT(lms_to_vec(*ret_or_error),
              ElementsAre(Pair(FloatEq(0), FloatEq(1 / 3.)),
                          Pair(FloatEq(0), FloatEq(1 / 3.)),
                          Pair(FloatEq(0), FloatEq(1 / 3.))));
}

TEST(RefineLandmarksFromHeatmapTest, KeepIfNotSure) {
  float z = -10000000000000000;
  // clang-format off
  std::vector<float> hm = CHW_to_HWC({
    z, z, z,
    0, z, z,
    z, z, z,
    z, z, z,
    0, z, z,
    z, z, z,
    z, z, z,
    0, z, z,
    z, z, z}, 3, 3, 3);
  // clang-format on

  auto ret_or_error = RefineLandmarksFromHeatMap(
      vec_to_lms({{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}), hm.data(), {3, 3, 3}, 3,
      0.6, true, true);
  MP_EXPECT_OK(ret_or_error);
  EXPECT_THAT(lms_to_vec(*ret_or_error),
              ElementsAre(Pair(FloatEq(0.5), FloatEq(0.5)),
                          Pair(FloatEq(0.5), FloatEq(0.5)),
                          Pair(FloatEq(0.5), FloatEq(0.5))));
}

TEST(RefineLandmarksFromHeatmapTest, Border) {
  float z = -10000000000000000;
  // clang-format off
  std::vector<float> hm = CHW_to_HWC({
    z, z, z,
    0, z, 0,
    z, z, z,

    z, z, z,
    0, z, 0,
    z, z, 0}, 3, 3, 2);
  // clang-format on

  auto ret_or_error =
      RefineLandmarksFromHeatMap(vec_to_lms({{0.0, 0.0}, {0.9, 0.9}}),
                                 hm.data(), {3, 3, 2}, 3, 0.1, true, true);
  MP_EXPECT_OK(ret_or_error);
  EXPECT_THAT(lms_to_vec(*ret_or_error),
              ElementsAre(Pair(FloatEq(0), FloatEq(1 / 3.)),
                          Pair(FloatEq(2 / 3.), FloatEq(1 / 6. + 2 / 6.))));
}

}  // namespace
}  // namespace mediapipe
