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

#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.pb.h"
#include "mediapipe/framework/port/integral_types.h"
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
  ::mediapipe::Status AddObservation(int position, const uint64 time_us);
  // Get the predicted position at a time.
  ::mediapipe::Status UpdatePrediction(const int64 time_us);
  // Get the state at a time.
  ::mediapipe::Status GetState(int* position);

 private:
  // Tuning options.
  KinematicOptions options_;
  // Min and max value the state can be.
  const int min_location_;
  const int max_location_;
  bool initialized_;
  float pixels_per_degree_;
  // Current state values.
  double current_position_px_;
  double current_velocity_deg_per_s_;
  uint64 current_time_;
};

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UNIFORM_ACCELERATION_PATH_SOLVER_H_
