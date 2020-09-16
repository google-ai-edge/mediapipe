#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"

namespace mediapipe {
namespace autoflip {

::mediapipe::Status KinematicPathSolver::AddObservation(int position,
                                                        const uint64 time_us) {
  if (!initialized_) {
    current_position_px_ = position;
    current_time_ = time_us;
    initialized_ = true;
    current_velocity_deg_per_s_ = 0;
    RET_CHECK_GT(pixels_per_degree_, 0)
        << "pixels_per_degree must be larger than 0.";
    RET_CHECK_GE(options_.update_rate_seconds(), 0)
        << "update_rate_seconds must be greater than 0.";
    RET_CHECK_GE(options_.min_motion_to_reframe(), options_.reframe_window())
        << "Reframe window cannot exceed min_motion_to_reframe.";
    return ::mediapipe::OkStatus();
  }

  RET_CHECK(current_time_ < time_us)
      << "Observation added before a prior observations.";

  double delta_degs = (position - current_position_px_) / pixels_per_degree_;

  // If the motion is smaller than the min, don't use the update.
  if (abs(delta_degs) < options_.min_motion_to_reframe()) {
    position = current_position_px_;
    delta_degs = 0;
  } else if (delta_degs > 0) {
    // Apply new position, less the reframe window size.
    position = position - pixels_per_degree_ * options_.reframe_window();
    delta_degs = (position - current_position_px_) / pixels_per_degree_;
  } else {
    // Apply new position, plus the reframe window size.
    position = position + pixels_per_degree_ * options_.reframe_window();
    delta_degs = (position - current_position_px_) / pixels_per_degree_;
  }

  // Time and position updates.
  double delta_t = (time_us - current_time_) / 1000000.0;

  // Observed velocity and then weighted update of this velocity.
  double observed_velocity = delta_degs / delta_t;
  double update_rate = std::min(delta_t / options_.update_rate_seconds(),
                                options_.max_update_rate());
  double updated_velocity = current_velocity_deg_per_s_ * (1 - update_rate) +
                            observed_velocity * update_rate;
  // Limited current velocity.
  current_velocity_deg_per_s_ =
      updated_velocity > 0 ? fmin(updated_velocity, options_.max_velocity())
                           : fmax(updated_velocity, -options_.max_velocity());

  // Update prediction based on time input.
  return UpdatePrediction(time_us);
}

::mediapipe::Status KinematicPathSolver::UpdatePrediction(const int64 time_us) {
  RET_CHECK(current_time_ < time_us)
      << "Prediction time added before a prior observation or prediction.";
  // Time since last state/prediction update.
  double delta_t = (time_us - current_time_) / 1000000.0;

  // Position update limited by min/max.

  const double update_position_px =
      current_position_px_ +
      current_velocity_deg_per_s_ * delta_t * pixels_per_degree_;
  if (update_position_px < min_location_) {
    current_position_px_ = min_location_;
    current_velocity_deg_per_s_ = 0;
  } else if (update_position_px > max_location_) {
    current_position_px_ = max_location_;
    current_velocity_deg_per_s_ = 0;
  } else {
    current_position_px_ = update_position_px;
  }
  current_time_ = time_us;

  return ::mediapipe::OkStatus();
}

::mediapipe::Status KinematicPathSolver::GetState(int* position) {
  RET_CHECK(initialized_) << "GetState called before first observation added.";
  *position = round(current_position_px_);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status KinematicPathSolver::UpdatePixelsPerDegree(
    const float pixels_per_degree) {
  RET_CHECK_GT(pixels_per_degree_, 0)
      << "pixels_per_degree must be larger than 0.";
  pixels_per_degree_ = pixels_per_degree;
  return ::mediapipe::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
