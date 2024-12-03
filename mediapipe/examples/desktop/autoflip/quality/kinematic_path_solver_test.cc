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

#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"

#include <cstdint>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

constexpr int64_t kMicroSecInSec = 1000000;
constexpr float kWidthFieldOfView = 60;

namespace mediapipe {
namespace autoflip {
namespace {

TEST(KinematicPathSolverTest, FailZeroPixelsPerDegree) {
  KinematicOptions options;
  KinematicPathSolver solver(options, 0, 1000, 0);
  EXPECT_FALSE(solver.AddObservation(500, kMicroSecInSec * 0).ok());
}

TEST(KinematicPathSolverTest, FailNotInitializedState) {
  KinematicOptions options;
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  EXPECT_FALSE(solver.GetState(&state).ok());
}

TEST(KinematicPathSolverTest, FailNotInitializedPrediction) {
  KinematicOptions options;
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int64_t timestamp = 0;
  EXPECT_FALSE(solver.UpdatePrediction(timestamp).ok());
}

TEST(KinematicPathSolverTest, PassNotEnoughMotionLargeImg) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(2.0);
  options.set_update_rate(1);
  options.set_max_velocity(1000);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_FLOAT_EQ(state, 500);
}

TEST(KinematicPathSolverTest, PassNotEnoughMotionSmallImg) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(2.0);
  options.set_update_rate(1);
  options.set_max_velocity(500);
  // Set degrees / pixel to 8.3
  KinematicPathSolver solver(options, 0, 500, 500.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(400, kMicroSecInSec * 0));
  // Move target by 10px / 8.3 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(410, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_FLOAT_EQ(state, 400);
}

TEST(KinematicPathSolverTest, PassEnoughMotionFiltered) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1);
  options.set_max_velocity(1000);
  options.set_filtering_time_window_us(3000000);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 2));
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 3));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_FLOAT_EQ(state, 500);
}

TEST(KinematicPathSolverTest, PassEnoughMotionNotFiltered) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1);
  options.set_max_velocity(1000);
  options.set_filtering_time_window_us(0);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 2));
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 3));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_FLOAT_EQ(state, 506.4);
}

TEST(KinematicPathSolverTest, PassEnoughMotionLargeImg) {
  KinematicOptions options;
  // Set min motion to 1deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.0000001);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(1000);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_FLOAT_EQ(state, 520);
}

TEST(KinematicPathSolverTest, PassEnoughMotionSmallImg) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.0000001);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(18);
  // Set degrees / pixel to 8.3
  KinematicPathSolver solver(options, 0, 500, 500.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(400, kMicroSecInSec * 0));
  // Move target by 10px / 8.3 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(410, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_FLOAT_EQ(state, 410);
}

TEST(KinematicPathSolverTest, FailReframeWindowSetting) {
  KinematicOptions options;
  // Set min motion to 1deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1);
  options.set_max_velocity(1000);
  // Set reframe window size to .75 for test.
  options.set_reframe_window(1.1);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  ASSERT_FALSE(solver.AddObservation(500, kMicroSecInSec * 0).ok());
}

TEST(KinematicPathSolverTest, PassReframeWindow) {
  KinematicOptions options;
  // Set min motion to 1deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.0000001);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(1000);
  // Set reframe window size to .75 for test.
  options.set_reframe_window(0.75);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move 1.2-.75 deg, * 16.6 = 7.47px + 500 =
  EXPECT_FLOAT_EQ(state, 507.5);
}

TEST(KinematicPathSolverTest, PassReframeWindowLowerUpper) {
  KinematicOptions options;
  // Set min motion to 1deg
  options.set_min_motion_to_reframe_upper(1.3);
  options.set_min_motion_to_reframe_lower(1.0);
  options.set_update_rate_seconds(.0000001);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(1000);
  // Set reframe window size to .75 for test.
  options.set_reframe_window(0.75);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move
  EXPECT_FLOAT_EQ(state, 500);
  MP_ASSERT_OK(solver.AddObservation(480, kMicroSecInSec * 2));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move
  EXPECT_FLOAT_EQ(state, 492.5);
}

TEST(KinematicPathSolverTest, PassCheckState) {
  KinematicOptions options;
  // Set min motion to 1deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.0000001);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(1000);
  // Set reframe window size to .75 for test.
  options.set_reframe_window(0.75);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  bool motion_state;
  MP_ASSERT_OK(
      solver.PredictMotionState(520, kMicroSecInSec * 1, &motion_state));
  EXPECT_TRUE(motion_state);
}

TEST(KinematicPathSolverTest, PassUpdateRate30FPS) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.25);
  options.set_max_update_rate(0.8);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1 / 30));
  MP_ASSERT_OK(solver.GetState(&state));
  // (0.033 / .25) * 20 =
  EXPECT_FLOAT_EQ(state, 502.6667);
}

TEST(KinematicPathSolverTest, PassUpdateRate10FPS) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.25);
  options.set_max_update_rate(0.8);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1 / 10));
  MP_ASSERT_OK(solver.GetState(&state));
  // (0.1 / .25) * 20 =
  EXPECT_FLOAT_EQ(state, 508);
}

TEST(KinematicPathSolverTest, PassUpdateRate) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(4);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int target_position;
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.GetTargetPosition(&target_position));
  EXPECT_EQ(target_position, 500);
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetTargetPosition(&target_position));
  EXPECT_EQ(target_position, 520);
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 505);
}

TEST(KinematicPathSolverTest, PassUpdateRateResolutionChange) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(4);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int target_position;
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.GetTargetPosition(&target_position));
  EXPECT_EQ(target_position, 500);
  MP_ASSERT_OK(solver.UpdateMinMaxLocation(0, 500));
  MP_ASSERT_OK(solver.UpdatePixelsPerDegree(500.0 / kWidthFieldOfView));
  MP_ASSERT_OK(solver.AddObservation(520 * 0.5, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetTargetPosition(&target_position));
  EXPECT_EQ(target_position, 520 * 0.5);
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 252.5);
}

TEST(KinematicPathSolverTest, PassMaxVelocityInt) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1.0);
  options.set_max_velocity(6);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(1000, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_EQ(state, 600);
}

TEST(KinematicPathSolverTest, PassMaxVelocity) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1.0);
  options.set_max_velocity(6);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(1000, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 600);
}

TEST(KinematicPathSolverTest, PassMaxVelocityScale) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1.0);
  options.set_max_velocity_scale(0.4);
  options.set_max_velocity_shift(-2.0);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(1000, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 666.6667);
}

TEST(KinematicPathSolverTest, PassDegPerPxChange) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(2.0);
  options.set_update_rate(1);
  options.set_max_velocity(1000);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_FLOAT_EQ(state, 500);
  MP_ASSERT_OK(solver.UpdatePixelsPerDegree(500.0 / kWidthFieldOfView));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 2));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_FLOAT_EQ(state, 516);
}

TEST(KinematicPathSolverTest, NoTimestampSmoothing) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1.0);
  options.set_max_velocity(6);
  options.set_mean_period_update_rate(1.0);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, 0));
  MP_ASSERT_OK(solver.AddObservation(1000, 1000000));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 600);
  MP_ASSERT_OK(solver.AddObservation(1000, 2200000));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 720);
}

TEST(KinematicPathSolverTest, TimestampSmoothing) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate(1.0);
  options.set_max_velocity(6);
  options.set_mean_period_update_rate(0.05);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(500, 0));
  MP_ASSERT_OK(solver.AddObservation(1000, 1000000));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 600);
  MP_ASSERT_OK(solver.AddObservation(1000, 2200000));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 701);
}

TEST(KinematicPathSolverTest, PassSetPosition) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.0000001);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(18);
  // Set degrees / pixel to 8.3
  KinematicPathSolver solver(options, 0, 500, 500.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(400, kMicroSecInSec * 0));
  // Move target by 10px / 8.3 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(410, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_FLOAT_EQ(state, 410);
  MP_ASSERT_OK(solver.SetState(400));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 400);
  // Expect to stay in bounds.
  MP_ASSERT_OK(solver.SetState(600));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 500);
  MP_ASSERT_OK(solver.SetState(-100));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 0);
}
TEST(KinematicPathSolverTest, PassBorderTest) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_max_update_rate(0.25);
  options.set_max_velocity_scale(0.5);
  options.set_max_velocity_shift(-1.0);

  KinematicPathSolver solver(options, 0, 500, 500.0 / kWidthFieldOfView);
  float state;
  MP_ASSERT_OK(solver.AddObservation(400, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(800, kMicroSecInSec * 0.1));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_FLOAT_EQ(state, 404.56668);
}

TEST(KinematicPathSolverTest, PassUpdateUpdateMinMaxLocationIfUninitialized) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(2.0);
  options.set_max_velocity(1000);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  MP_EXPECT_OK(solver.UpdateMinMaxLocation(0, 500));
}

TEST(KinematicPathSolverTest, PassUpdateUpdateMinMaxLocationIfInitialized) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(2.0);
  options.set_max_velocity(1000);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_EXPECT_OK(solver.UpdateMinMaxLocation(0, 500));
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
