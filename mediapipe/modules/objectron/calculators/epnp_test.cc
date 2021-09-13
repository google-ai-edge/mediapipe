#include "mediapipe/modules/objectron/calculators/epnp.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/tool/test_util.h"

namespace mediapipe {
namespace {

using Eigen::AngleAxisf;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::Matrix4f;
using Eigen::RowMajor;
using Eigen::Vector2f;
using Eigen::Vector3f;
using ::testing::HasSubstr;
using ::testing::Test;
using ::testing::status::StatusIs;
using Matrix3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

constexpr uint8_t kNumKeypoints = 9;

// clang-format off
constexpr float kUnitBox[] = { 0.0f,  0.0f,  0.0f,
                              -0.5f, -0.5f, -0.5f,
                              -0.5f, -0.5f,  0.5f,
                              -0.5f,  0.5f, -0.5f,
                              -0.5f,  0.5f,  0.5f,
                               0.5f, -0.5f, -0.5f,
                               0.5f, -0.5f,  0.5f,
                               0.5f,  0.5f, -0.5f,
                               0.5f,  0.5f,  0.5f, };
// clang-format on

constexpr float kFocalX = 1.0f;
constexpr float kFocalY = 1.0f;
constexpr float kCenterX = 0.0f;
constexpr float kCenterY = 0.0f;

constexpr float kAzimuth = 90.0f * M_PI / 180.0f;
constexpr float kElevation = 45.0f * M_PI / 180.0f;
constexpr float kTilt = 15.0f * M_PI / 180.0f;

constexpr float kTranslationArray[] = {0.0f, 0.0f, -100.0f};

constexpr float kScaleArray[] = {50.0f, 50.0f, 50.0f};

class SolveEpnpTest : public Test {
 protected:
  SolveEpnpTest() {}

  void SetUp() override {
    // Create vertices in world frame.
    Map<const Matrix<float, kNumKeypoints, 3, RowMajor>> vertices_w(kUnitBox);

    // Create Pose.
    Matrix3f rotation;
    rotation = AngleAxisf(kTilt, Vector3f::UnitZ()) *
               AngleAxisf(kElevation, Vector3f::UnitX()) *
               AngleAxisf(kAzimuth, Vector3f::UnitY());
    Map<const Vector3f> translation(kTranslationArray);
    Map<const Vector3f> scale(kScaleArray);

    // Generate 3d vertices in camera frame.
    const auto vertices_c =
        ((rotation * scale.asDiagonal() * vertices_w.transpose()).colwise() +
         translation)
            .transpose();

    // Generate input 2d points.
    std::vector<Vector2f> input_2d_points;
    std::vector<Vector3f> expected_3d_points;
    for (int i = 0; i < kNumKeypoints; ++i) {
      const auto x = vertices_c(i, 0);
      const auto y = vertices_c(i, 1);
      const auto z = vertices_c(i, 2);

      const float x_ndc = -kFocalX * x / z + kCenterX;
      const float y_ndc = -kFocalY * y / z + kCenterY;

      const float x_pixel = (1.0f + x_ndc) / 2.0f;
      const float y_pixel = (1.0f - y_ndc) / 2.0f;

      expected_3d_points_.emplace_back(x, y, z);
      input_2d_points_.emplace_back(x_pixel, y_pixel);
    }
  }

  void VerifyOutput3dPoints(const std::vector<Vector3f>& output_3d_points) {
    EXPECT_EQ(kNumKeypoints, output_3d_points.size());
    const float scale = output_3d_points[0].z() / expected_3d_points_[0].z();
    for (int i = 0; i < kNumKeypoints; ++i) {
      EXPECT_NEAR(output_3d_points[i].x(), expected_3d_points_[i].x() * scale,
                  2.e-6f);
      EXPECT_NEAR(output_3d_points[i].y(), expected_3d_points_[i].y() * scale,
                  2.e-6f);
      EXPECT_NEAR(output_3d_points[i].z(), expected_3d_points_[i].z() * scale,
                  2.e-6f);
    }
  }

  std::vector<Vector2f> input_2d_points_;
  std::vector<Vector3f> expected_3d_points_;
};

TEST_F(SolveEpnpTest, SolveEpnp) {
  std::vector<Vector3f> output_3d_points;
  MP_ASSERT_OK(SolveEpnp(kFocalX, kFocalY, kCenterX, kCenterY,
                         /*portrait*/ false, input_2d_points_,
                         &output_3d_points));
  // Test output 3D points.
  VerifyOutput3dPoints(output_3d_points);
}

TEST_F(SolveEpnpTest, SolveEpnppPortrait) {
  std::vector<Vector3f> output_3d_points;
  MP_ASSERT_OK(SolveEpnp(kFocalX, kFocalY, kCenterX, kCenterY,
                         /*portrait*/ true, input_2d_points_,
                         &output_3d_points));
  // Test output 3D points.
  for (auto& point_3d : output_3d_points) {
    const auto x = point_3d.x();
    const auto y = point_3d.y();
    // Convert from portrait mode to normal mode, y => x, x => -y.
    point_3d.x() = y;
    point_3d.y() = -x;
  }
  VerifyOutput3dPoints(output_3d_points);
}

TEST_F(SolveEpnpTest, SolveEpnpProjectionMatrix) {
  Matrix4f projection_matrix;
  // clang-format off
  projection_matrix << kFocalX,    0.0f, kCenterX, 0.0f,
                          0.0f, kFocalY, kCenterY, 0.0f,
                          0.0f,    0.0f,    -1.0f, 0.0f,
                          0.0f,    0.0f,    -1.0f, 0.0f;
  // clang-format on

  std::vector<Vector3f> output_3d_points;
  MP_ASSERT_OK(SolveEpnp(projection_matrix, /*portrait*/ false,
                         input_2d_points_, &output_3d_points));

  // Test output 3D points.
  VerifyOutput3dPoints(output_3d_points);
}

TEST_F(SolveEpnpTest, BadInput2dPoints) {
  // Generate empty input 2D points.
  std::vector<Vector2f> input_2d_points;
  std::vector<Vector3f> output_3d_points;
  EXPECT_THAT(SolveEpnp(kFocalX, kFocalY, kCenterX, kCenterY,
                        /*portrait*/ false, input_2d_points, &output_3d_points),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input must has")));
}

TEST_F(SolveEpnpTest, BadOutput3dPoints) {
  // Generate null output 3D points.
  std::vector<Vector3f>* output_3d_points = nullptr;
  EXPECT_THAT(SolveEpnp(kFocalX, kFocalY, kCenterX, kCenterY,
                        /*portrait*/ false, input_2d_points_, output_3d_points),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Output pointer output_points_3d is Null."));
}

}  // namespace
}  // namespace mediapipe
