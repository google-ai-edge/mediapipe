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

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

constexpr int64 kMicroSecInSec = 1000000;
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
  int state;
  EXPECT_FALSE(solver.GetState(&state).ok());
}

TEST(KinematicPathSolverTest, FailNotInitializedPrediction) {
  KinematicOptions options;
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int64 timestamp = 0;
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
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_EQ(state, 500);
}

TEST(KinematicPathSolverTest, PassNotEnoughMotionSmallImg) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(2.0);
  options.set_update_rate(1);
  options.set_max_velocity(500);
  // Set degrees / pixel to 8.3
  KinematicPathSolver solver(options, 0, 500, 500.0 / kWidthFieldOfView);
  int state;
  MP_ASSERT_OK(solver.AddObservation(400, kMicroSecInSec * 0));
  // Move target by 10px / 8.3 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(410, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_EQ(state, 400);
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
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_EQ(state, 520);
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
  int state;
  MP_ASSERT_OK(solver.AddObservation(400, kMicroSecInSec * 0));
  // Move target by 10px / 8.3 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(410, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_EQ(state, 410);
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
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move 1.2-.75 deg, * 16.6 = 7.47px + 500 =
  EXPECT_EQ(state, 507);
}

TEST(KinematicPathSolverTest, PassUpdateRate30FPS) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.25);
  options.set_max_update_rate(0.8);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1 / 30));
  MP_ASSERT_OK(solver.GetState(&state));
  // (0.033 / .25) * 20 =
  EXPECT_EQ(state, 503);
}

TEST(KinematicPathSolverTest, PassUpdateRate10FPS) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(.25);
  options.set_max_update_rate(0.8);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1 / 10));
  MP_ASSERT_OK(solver.GetState(&state));
  // (0.1 / .25) * 20 =
  EXPECT_EQ(state, 508);
}

TEST(KinematicPathSolverTest, PassUpdateRate) {
  KinematicOptions options;
  options.set_min_motion_to_reframe(1.0);
  options.set_update_rate_seconds(4);
  options.set_max_update_rate(1.0);
  options.set_max_velocity(18);
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  EXPECT_EQ(state, 505);
}

TEST(KinematicPathSolverTest, PassMaxVelocity) {
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

TEST(KinematicPathSolverTest, PassDegPerPxChange) {
  KinematicOptions options;
  // Set min motion to 2deg
  options.set_min_motion_to_reframe(2.0);
  options.set_update_rate(1);
  options.set_max_velocity(1000);
  // Set degrees / pixel to 16.6
  KinematicPathSolver solver(options, 0, 1000, 1000.0 / kWidthFieldOfView);
  int state;
  MP_ASSERT_OK(solver.AddObservation(500, kMicroSecInSec * 0));
  // Move target by 20px / 16.6 = 1.2deg
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 1));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to not move.
  EXPECT_EQ(state, 500);
  MP_ASSERT_OK(solver.UpdatePixelsPerDegree(500.0 / kWidthFieldOfView));
  MP_ASSERT_OK(solver.AddObservation(520, kMicroSecInSec * 2));
  MP_ASSERT_OK(solver.GetState(&state));
  // Expect cam to move.
  EXPECT_EQ(state, 516);
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
