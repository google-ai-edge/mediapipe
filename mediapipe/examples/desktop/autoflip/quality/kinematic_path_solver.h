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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UNIFORM_ACCELERATION_PATH_SOLVER_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UNIFORM_ACCELERATION_PATH_SOLVER_H_

#include <cstdint>
#include <deque>

#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// Kinematic path solver class is a stateful 1d position estimator based loosely
// on a differential kalman filter that is specifically designed to control a
// camera.  It utilizes a Kalman filters predict/update interface for estimating
// the best camera focus position and updating that estimate when a measurement
// is available.  Tuning controls include: update_rate: how much to update the
// existing state with a new state. max_velocity: max speed of the state per
// second. min_motion_to_reframe: only updating the state if a measurement
// exceeds this threshold.
class KinematicPathSolver {
 public:
  KinematicPathSolver(const KinematicOptions& options, const int min_location,
                      const int max_location, float pixels_per_degree)
      : options_(options),
        min_location_(min_location),
        max_location_(max_location),
        initialized_(false),
        pixels_per_degree_(pixels_per_degree) {}
  // Add an observation (detection) at a position and time.
  absl::Status AddObservation(int position, const uint64_t time_us);
  // Get the predicted position at a time.
  absl::Status UpdatePrediction(const int64_t time_us);
  // Get the state at a time, as an int.
  absl::Status GetState(int* position);
  // Get the state at a time, as a float.
  absl::Status GetState(float* position);
  // Overwrite the current state value.
  absl::Status SetState(const float position);
  // Update PixelPerDegree value.
  absl::Status UpdatePixelsPerDegree(const float pixels_per_degree);
  // Provide the current target position of the reframe action.
  absl::Status GetTargetPosition(int* target_position);
  // Change min/max location and update state based on new scaling.
  absl::Status UpdateMinMaxLocation(const int min_location,
                                    const int max_location);
  // Check if motion is within the reframe window, return false if not.
  bool IsMotionTooSmall(double delta_degs);
  // Check if a position measurement will cause the camera to be in motion
  // without updating the internal state.
  absl::Status PredictMotionState(int position, const uint64_t time_us,
                                  bool* state);
  // Clear any history buffer of positions that are used when
  // filtering_time_window_us is set to a non-zero value.
  void ClearHistory();
  // Provides the change in position from last state.
  absl::Status GetDeltaState(float* delta_position);

  bool IsInitialized() { return initialized_; }

 private:
  // Tuning options.
  KinematicOptions options_;
  // Min and max value the state can be.
  int min_location_;
  int max_location_;
  bool initialized_;
  float pixels_per_degree_;
  // Current state values.
  double current_position_px_;
  double prior_position_px_;
  double current_velocity_deg_per_s_;
  uint64_t current_time_ = 0;
  // History of observations (second) and their time (first).
  std::deque<std::pair<uint64_t, int>> raw_positions_at_time_;
  // Current target position.
  double target_position_px_;
  // Defines if the camera is moving to a target (true) or reached a target
  // within a tolerance (false).
  bool motion_state_;
  // Average period of incoming frames.
  double mean_delta_t_;
};

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UNIFORM_ACCELERATION_PATH_SOLVER_H_
