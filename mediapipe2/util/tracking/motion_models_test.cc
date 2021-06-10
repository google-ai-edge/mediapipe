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

#include "mediapipe/util/tracking/motion_models.h"

#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/tracking/motion_estimation.h"

namespace mediapipe {
namespace {

static const float kArrayFloat[] = {1, 2, 3, 4, 5, 6, 7, 8};
static const double kArrayDouble[] = {1, 2, 3, 4, 5, 6, 7, 8};

class MotionModelsTest : public ::testing::Test {};

// Test from/to array and parameter indexing functions.
template <class Model>
void CheckFromArrayAndGetParameter(const char* model_zero_string,
                                   const char* model_identity_string) {
  Model model_zero;
  Model model_identity;

  ASSERT_TRUE(
      proto_ns::TextFormat::ParseFromString(model_zero_string, &model_zero));
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(model_identity_string,
                                                    &model_identity));

  typedef ModelAdapter<Model> Adapter;

  EXPECT_THAT(Adapter::FromFloatPointer(kArrayFloat, false),
              mediapipe::EqualsProto(model_zero));

  EXPECT_THAT(Adapter::FromFloatPointer(kArrayFloat, true),
              mediapipe::EqualsProto(model_identity));

  EXPECT_THAT(Adapter::FromDoublePointer(kArrayDouble, false),
              mediapipe::EqualsProto(model_zero));

  EXPECT_THAT(Adapter::FromDoublePointer(kArrayDouble, true),
              mediapipe::EqualsProto(model_identity));

  ASSERT_LE(Adapter::NumParameters(), 8);
  for (int i = 0; i < Adapter::NumParameters(); ++i) {
    EXPECT_EQ(kArrayFloat[i], Adapter::GetParameter(model_zero, i));
  }
}

TEST_F(MotionModelsTest, FromArrayAndGetParameter) {
  CheckFromArrayAndGetParameter<TranslationModel>("dx: 1 dy: 2", "dx: 1 dy: 2");

  CheckFromArrayAndGetParameter<SimilarityModel>(
      "dx: 1 dy: 2 scale: 3 rotation: 4", "dx: 1 dy: 2 scale: 4 rotation: 4");

  CheckFromArrayAndGetParameter<LinearSimilarityModel>("dx: 1 dy: 2 a: 3 b: 4",
                                                       "dx: 1 dy: 2 a: 4 b: 4");

  CheckFromArrayAndGetParameter<AffineModel>("dx: 1 dy: 2 a: 3 b: 4 c: 5 d: 6",
                                             "dx: 1 dy: 2 a: 4 b: 4 c: 5 d: 7");

  CheckFromArrayAndGetParameter<Homography>(
      "h_00: 1 h_01: 2 h_02: 3 "
      "h_10: 4 h_11: 5 h_12: 6 "
      "h_20: 7 h_21: 8         ",
      "h_00: 2 h_01: 2 h_02: 3 "
      "h_10: 4 h_11: 6 h_12: 6 "
      "h_20: 7 h_21: 8         ");
}

// Test point transformations.
template <class Model>
void CheckTransformPoint(const char* model_string, float x_in, float y_in,
                         float x_out, float y_out) {
  Model model;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(model_string, &model));

  const Vector2_f output =
      ModelAdapter<Model>::TransformPoint(model, Vector2_f(x_in, y_in));

  EXPECT_NEAR(x_out, output.x(), 1e-5);
  EXPECT_NEAR(y_out, output.y(), 1e-5);
}

TEST_F(MotionModelsTest, TransformPoint) {
  CheckTransformPoint<TranslationModel>("dx: 0 dy: 0", 1, 1, 1, 1);
  CheckTransformPoint<TranslationModel>("dx: 1 dy: -1", 1, 1, 2, 0);

  CheckTransformPoint<SimilarityModel>(
      "dx: 0 dy: 0 scale: 1 rotation: 1.57079633", 1, 2, -2, 1);
  CheckTransformPoint<SimilarityModel>(
      "dx: 1 dy: -1 scale: 1 rotation: 1.57079633", 1, 2, -1, 0);
  CheckTransformPoint<SimilarityModel>(
      "dx: 1 dy: -1 scale: 2 rotation: 1.57079633", 1, 2, -3, 1);

  CheckTransformPoint<LinearSimilarityModel>("dx: 0 dy: 0 a: 1 b: -0.5", 1, 2,
                                             2, 1.5);
  CheckTransformPoint<LinearSimilarityModel>("dx: 0.5 dy: -0.5 a: 1 b: 0.5", 1,
                                             2, 0.5, 2);
  CheckTransformPoint<LinearSimilarityModel>("dx: 0.5 dy: -0.5 a: 0.5 b: 0.5",
                                             1, 2, 0, 1);

  CheckTransformPoint<AffineModel>("dx: 0 dy: 0 a: 1 b: 0.5 c: -0.5 d: 1", 1, 2,
                                   2, 1.5);
  CheckTransformPoint<AffineModel>("dx: 0.5 dy: -0.5 a: 2 b: -0.5 c: 0.5 d: 1",
                                   1, 2, 1.5, 2);
  CheckTransformPoint<AffineModel>("dx: 1 dy: -1 a: 2 b: -2 c: 1 d: -1", 1, 2,
                                   -1, -2);

  // Transformations by Homography are followed by divsion by the 3rd element.
  // Test division by value != 1.
  CheckTransformPoint<Homography>(
      "h_00: 1  h_01: 2  h_02: 3 "
      "h_10: 4  h_11: 3  h_12: 6 "
      "h_20: 7  h_21: 8          ",
      1, 2, 8.0 / 24.0, 16.0 / 24.0);
  // Test division by 1.
  CheckTransformPoint<Homography>(
      "h_00: 1  h_01:  2  h_02: 3 "
      "h_10: 4  h_11:  3  h_12: 6 "
      "h_20: 2  h_21: -1          ",
      1, 2, 8.0, 16.0);
}

// Test model inversions.
template <class Model>
void CheckInvert(const char* model_string, const char* inv_model_string) {
  Model model;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(model_string, &model));

  Model inv_model_expected;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(inv_model_string,
                                                    &inv_model_expected));

  typedef ModelAdapter<Model> Adapter;
  const Model inv_model_actual = Adapter::Invert(model);

  for (int i = 0; i < Adapter::NumParameters(); ++i) {
    EXPECT_NEAR(Adapter::GetParameter(inv_model_expected, i),
                Adapter::GetParameter(inv_model_actual, i), 1e-5)
        << "Parameter index: " << i << " of total " << Adapter::NumParameters();
  }
}

TEST_F(MotionModelsTest, Invert) {
  CheckInvert<TranslationModel>("dx:  1 dy: -2", "dx: -1 dy:  2");

  CheckInvert<SimilarityModel>("dx: 0 dy:  0 scale: 1   rotation:  1.57079633",
                               "dx: 0 dy:  0 scale: 1   rotation: -1.57079633");
  CheckInvert<SimilarityModel>("dx: 1 dy: -2 scale: 1   rotation:  1.57079633",
                               "dx: 2 dy:  1 scale: 1   rotation: -1.57079633");
  CheckInvert<SimilarityModel>("dx: 1 dy: -2 scale: 0.5 rotation:  1.57079633",
                               "dx: 4 dy:  2 scale: 2   rotation: -1.57079633");

  CheckInvert<LinearSimilarityModel>("dx:  1    dy:  2    a: 3    b:  4    ",
                                     "dx: -0.44 dy: -0.08 a: 0.12 b: -0.16 ");

  // Test division by value != 1.
  CheckInvert<Homography>(
      "h_00:  1     h_01:  2     h_02:  3 "
      "h_10: -3     h_11: -2     h_12: -1 "
      "h_20:  8     h_21: -1              ",
      "h_00: -0.75  h_01: -1.25  h_02:  1 "
      "h_10: -1.25  h_11: -5.75  h_12: -2 "
      "h_20:  4.75  h_21:  4.25           ");
  // Test division by 1.
  CheckInvert<Homography>(
      "h_00: -0.75  h_01: -1.25  h_02:  1 "
      "h_10: -1.25  h_11: -5.75  h_12: -2 "
      "h_20:  4.75  h_21:  4.25           ",
      "h_00:  1     h_01:  2     h_02:  3 "
      "h_10: -3     h_11: -2     h_12: -1 "
      "h_20:  8     h_21: -1              ");
}

// Test model compositions.
template <class Model>
void CheckCompose(const char* model_string1, const char* model_string2,
                  const char* composed_string) {
  Model model1;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(model_string1, &model1));

  Model model2;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(model_string2, &model2));

  Model composed_expected;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(composed_string,
                                                    &composed_expected));

  typedef ModelAdapter<Model> Adapter;
  const Model composed_actual = Adapter::Compose(model1, model2);

  for (int i = 0; i < Adapter::NumParameters(); ++i) {
    EXPECT_NEAR(Adapter::GetParameter(composed_expected, i),
                Adapter::GetParameter(composed_actual, i), 1e-5)
        << "Parameter index: " << i << " of total " << Adapter::NumParameters();
  }
}

TEST_F(MotionModelsTest, Compose) {
  CheckCompose<TranslationModel>("dx:  1 dy: -2", "dx: -3 dy:  4",
                                 "dx: -2 dy:  2");

  CheckCompose<SimilarityModel>(
      "dx:  1   dy:  2 scale: 0.5 rotation:  1.57079633 ",
      "dx: -2   dy: -1 scale: 2   rotation: -1.57079633 ",
      "dx:  1.5 dy:  1 scale: 1   rotation:  0          ");

  CheckCompose<LinearSimilarityModel>("dx:  1   dy:  2   a: 0.5  b:  0.5  ",
                                      "dx: -2   dy: -1   a: 2    b: -0.5  ",
                                      "dx:  0.5 dy:  0.5 a: 1.25 b:  0.75 ");

  // Test division by value != 1.
  CheckCompose<Homography>(
      "h_00:  1  h_01:  2    h_02:  3 "
      "h_10:  4  h_11:  5    h_12:  6 "
      "h_20:  1  h_21: -1             ",
      "h_00: -3  h_01: -2    h_02: -1 "
      "h_10: -4  h_11: -5    h_12: -2 "
      "h_20:  7  h_21:  8             ",
      "h_00:  5  h_01:  6    h_02: -1 "
      "h_10:  5  h_11:  7.5  h_12: -4 "
      "h_20:  4  h_21:  5.5           ");
  // Test division by 1.
  CheckCompose<Homography>(
      "h_00:  1  h_01:  2  h_02:  3 "
      "h_10:  4  h_11:  5  h_12:  6 "
      "h_20:  2  h_21: -1           ",
      "h_00: -3  h_01: -2  h_02: -1 "
      "h_10: -4  h_11: -5  h_12: -2 "
      "h_20:  7  h_21:  8           ",
      "h_00: 10  h_01: 12  h_02: -2 "
      "h_10: 10  h_11: 15  h_12: -8 "
      "h_20:  5  h_21:  9           ");
}

// Test conversions between models and their affine representations, and
// vice-versa.
template <class Model>
void CheckToFromAffine(const char* model_string, const char* affine_string) {
  Model model;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(model_string, &model));

  AffineModel affine;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(affine_string, &affine));

  typedef ModelAdapter<Model> Adapter;
  EXPECT_THAT(Adapter::ToAffine(model), mediapipe::EqualsProto(affine));
  EXPECT_THAT(Adapter::FromAffine(affine), mediapipe::EqualsProto(model));
}

TEST_F(MotionModelsTest, ToFromAffine) {
  CheckToFromAffine<TranslationModel>("dx: 1 dy: 2",
                                      "dx: 1 dy: 2 a: 1 b: 0 c: 0 d: 1");

  CheckToFromAffine<LinearSimilarityModel>("dx: 1 dy: 2 a: 3 b: -4",
                                           "dx: 1 dy: 2 a: 3 b: 4 c: -4 d: 3");

  CheckToFromAffine<AffineModel>("dx: 1 dy: 2 a: 3 b: 4 c: 5 d: 6",
                                 "dx: 1 dy: 2 a: 3 b: 4 c: 5 d: 6");

  CheckToFromAffine<Homography>(
      "h_00: 3  h_01: 4  h_02: 1 "
      "h_10: 5  h_11: 6  h_12: 2 "
      "h_20: 0  h_21: 0          ",
      "dx: 1 dy: 2 a: 3 b: 4 c: 5 d: 6");
  Homography homography;
  ASSERT_TRUE(
      proto_ns::TextFormat::ParseFromString("h_00: 3  h_01: 4  h_02: 1 "
                                            "h_10: 5  h_11: 6  h_12: 2 "
                                            "h_20: 0  h_21: 0          ",
                                            &homography));

  EXPECT_TRUE(HomographyAdapter::IsAffine(homography));
  homography.set_h_20(7);
  homography.set_h_21(8);
  EXPECT_FALSE(HomographyAdapter::IsAffine(homography));
}

TEST_F(MotionModelsTest, ProjectModels) {
  // Express models w.r.t. center for easy testing.
  LinearSimilarityModel center_trans =
      LinearSimilarityAdapter::FromArgs(50, 50, 1, 0);
  LinearSimilarityModel inv_center_trans =
      LinearSimilarityAdapter::FromArgs(-50, -50, 1, 0);

  // 20 x 10 translation with scaling of factor 2 and rotation.
  LinearSimilarityModel lin_sim =
      LinearSimilarityAdapter::FromArgs(20, 10, 2 * cos(0.2), 2 * sin(0.2));

  LinearSimilarityModel lin_sim_center =
      ModelCompose3(center_trans, lin_sim, inv_center_trans);

  TranslationModel translation =
      TranslationAdapter::ProjectFrom(lin_sim_center, 100, 100);
  EXPECT_NEAR(translation.dx(), 20, 1e-3);
  EXPECT_NEAR(translation.dy(), 10, 1e-3);

  translation = ProjectViaFit<TranslationModel>(lin_sim_center, 100, 100);
  EXPECT_NEAR(translation.dx(), 20, 1e-3);
  EXPECT_NEAR(translation.dy(), 10, 1e-3);

  Homography homog = HomographyAdapter::FromArgs(
      1, 0, 10, 0, 1, 20, 5e-3, 1e-3);  // Perspective transform: yaw + pitch.

  Homography homog_center =
      ModelCompose3(HomographyAdapter::Embed(center_trans), homog,
                    HomographyAdapter::Embed(inv_center_trans));
  // Rendering:
  // https://www.wolframalpha.com/input/?i=ListPlot%5B%7B+%7B7,-7%7D,+%7B108,16%7D,+%7B104,96%7D,+%7B12.5,+125%7D,+%7B7,-7%7D%5D

  translation = TranslationAdapter::ProjectFrom(homog_center, 100, 100);
  EXPECT_NEAR(translation.dx(), 10, 1e-3);
  EXPECT_NEAR(translation.dy(), 20, 1e-3);

  // TODO: Investigate how ProjectViaFit can yield similar result.
}

}  // namespace
}  // namespace mediapipe
