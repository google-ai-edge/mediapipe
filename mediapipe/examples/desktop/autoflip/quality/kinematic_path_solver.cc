#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"

constexpr float kMinVelocity = 0.5;

namespace mediapipe {
namespace autoflip {

namespace {
int Median(const std::deque<std::pair<uint64_t, int>>& positions_raw) {
  std::deque<int> positions;
  for (const auto& position : positions_raw) {
    positions.push_back(position.second);
  }

  size_t n = positions.size() / 2;
  nth_element(positions.begin(), positions.begin() + n, positions.end());
  return positions[n];
}
}  // namespace

bool KinematicPathSolver::IsMotionTooSmall(double delta_degs) {
  if (options_.has_min_motion_to_reframe()) {
    return abs(delta_degs) < options_.min_motion_to_reframe();
  } else if (delta_degs > 0) {
    return delta_degs < options_.min_motion_to_reframe_upper();
  } else {
    return abs(delta_degs) < options_.min_motion_to_reframe_lower();
  }
}

void KinematicPathSolver::ClearHistory() { raw_positions_at_time_.clear(); }

absl::Status KinematicPathSolver::PredictMotionState(int position,
                                                     const uint64_t time_us,
                                                     bool* state) {
  if (!initialized_) {
    *state = false;
    return absl::OkStatus();
  }

  auto raw_positions_at_time_copy = raw_positions_at_time_;

  raw_positions_at_time_copy.push_front(
      std::pair<uint64_t, int>(time_us, position));
  while (raw_positions_at_time_copy.size() > 1) {
    if (static_cast<int64_t>(raw_positions_at_time_copy.back().first) <
        static_cast<int64_t>(time_us) - options_.filtering_time_window_us()) {
      raw_positions_at_time_copy.pop_back();
    } else {
      break;
    }
  }

  int filtered_position = Median(raw_positions_at_time_copy);
  filtered_position =
      std::clamp(filtered_position, min_location_, max_location_);

  double delta_degs =
      (filtered_position - current_position_px_) / pixels_per_degree_;

  // If the motion is smaller than the min_motion_to_reframe and camera is
  // stationary, don't use the update.
  if (IsMotionTooSmall(delta_degs) && !motion_state_) {
    *state = false;
  } else if (abs(delta_degs) < options_.reframe_window() && motion_state_) {
    // If the motion is smaller than the reframe_window and camera is moving,
    // don't use the update.
    *state = false;
  } else if (prior_position_px_ == current_position_px_ && motion_state_) {
    // Camera isn't actually moving. Likely face is past bounds.
    *state = false;
  } else {
    // Apply new position, plus the reframe window size.
    *state = true;
  }

  return absl::OkStatus();
}

absl::Status KinematicPathSolver::AddObservation(int position,
                                                 const uint64_t time_us) {
  if (!initialized_) {
    if (position < min_location_) {
      current_position_px_ = min_location_;
    } else if (position > max_location_) {
      current_position_px_ = max_location_;
    } else {
      current_position_px_ = position;
    }
    target_position_px_ = position;
    prior_position_px_ = current_position_px_;
    motion_state_ = false;
    mean_delta_t_ = -1;
    raw_positions_at_time_.push_front(
        std::pair<uint64_t, int>(time_us, position));
    current_time_ = time_us;
    initialized_ = true;
    current_velocity_deg_per_s_ = 0;
    RET_CHECK_GT(pixels_per_degree_, 0)
        << "pixels_per_degree must be larger than 0.";
    RET_CHECK_GE(options_.update_rate_seconds(), 0)
        << "update_rate_seconds must be greater than 0.";
    RET_CHECK_GE(options_.filtering_time_window_us(), 0)
        << "update_rate_seconds must be greater than 0.";
    RET_CHECK_GE(options_.mean_period_update_rate(), 0)
        << "mean_period_update_rate must be greater than 0.";
    RET_CHECK(options_.has_min_motion_to_reframe() ^
              (options_.has_min_motion_to_reframe_upper() &&
               options_.has_min_motion_to_reframe_lower()))
        << "Must set min_motion_to_reframe or min_motion_to_reframe_upper and "
           "min_motion_to_reframe_lower.";
    if (options_.has_min_motion_to_reframe()) {
      RET_CHECK_GE(options_.min_motion_to_reframe(), options_.reframe_window())
          << "Reframe window cannot exceed min_motion_to_reframe.";
    } else {
      RET_CHECK_GE(options_.min_motion_to_reframe_upper(),
                   options_.reframe_window())
          << "Reframe window cannot exceed min_motion_to_reframe.";
      RET_CHECK_GE(options_.min_motion_to_reframe_lower(),
                   options_.reframe_window())
          << "Reframe window cannot exceed min_motion_to_reframe.";
    }
    RET_CHECK(options_.has_max_velocity() ^
              (options_.has_max_velocity_scale() &&
               options_.has_max_velocity_shift()))
        << "Must either set max_velocity or set both max_velocity_scale and "
           "max_velocity_shift.";
    return absl::OkStatus();
  }

  RET_CHECK(current_time_ < time_us)
      << "Observation added before a prior observations.";

  raw_positions_at_time_.push_front(
      std::pair<uint64_t, int>(time_us, position));
  while (raw_positions_at_time_.size() > 1) {
    if (static_cast<int64_t>(raw_positions_at_time_.back().first) <
        static_cast<int64_t>(time_us) - options_.filtering_time_window_us()) {
      raw_positions_at_time_.pop_back();
    } else {
      break;
    }
  }

  int filtered_position = Median(raw_positions_at_time_);

  float min_reframe = (options_.has_min_motion_to_reframe()
                           ? options_.min_motion_to_reframe()
                           : options_.min_motion_to_reframe_lower()) *
                      pixels_per_degree_;
  float max_reframe = (options_.has_min_motion_to_reframe()
                           ? options_.min_motion_to_reframe()
                           : options_.min_motion_to_reframe_upper()) *
                      pixels_per_degree_;

  filtered_position = fmax(min_location_ - min_reframe, filtered_position);
  filtered_position = fmin(max_location_ + max_reframe, filtered_position);

  double delta_degs =
      (filtered_position - current_position_px_) / pixels_per_degree_;

  double max_velocity =
      options_.has_max_velocity()
          ? options_.max_velocity()
          : fmax(abs(delta_degs * options_.max_velocity_scale()) +
                     options_.max_velocity_shift(),
                 kMinVelocity);

  // If the motion is smaller than the min_motion_to_reframe and camera is
  // stationary, don't use the update.
  if (IsMotionTooSmall(delta_degs) && !motion_state_) {
    delta_degs = 0;
    motion_state_ = false;
  } else if (abs(delta_degs) < options_.reframe_window() && motion_state_) {
    // If the motion is smaller than the reframe_window and camera is moving,
    // don't use the update.
    delta_degs = 0;
    motion_state_ = false;
  } else if (delta_degs > 0) {
    // Apply new position, less the reframe window size.
    target_position_px_ =
        filtered_position - pixels_per_degree_ * options_.reframe_window();
    delta_degs =
        (target_position_px_ - current_position_px_) / pixels_per_degree_;
    motion_state_ = true;
  } else {
    // Apply new position, plus the reframe window size.
    target_position_px_ =
        filtered_position + pixels_per_degree_ * options_.reframe_window();
    delta_degs =
        (target_position_px_ - current_position_px_) / pixels_per_degree_;
    motion_state_ = true;
  }

  // Time and position updates.
  double delta_t_sec = (time_us - current_time_) / 1000000.0;
  if (options_.max_delta_time_sec() > 0) {
    // If updates are very infrequent, then limit the max time difference.
    delta_t_sec = fmin(delta_t_sec, options_.max_delta_time_sec());
  }
  // Time since last state/prediction update, smoothed by
  // mean_period_update_rate.
  if (mean_delta_t_ < 0) {
    mean_delta_t_ = delta_t_sec;
  } else {
    mean_delta_t_ = mean_delta_t_ * (1 - options_.mean_period_update_rate()) +
                    delta_t_sec * options_.mean_period_update_rate();
  }

  // Observed velocity and then weighted update of this velocity (deg/sec).
  double observed_velocity = delta_degs / delta_t_sec;
  double update_rate = std::min(mean_delta_t_ / options_.update_rate_seconds(),
                                options_.max_update_rate());
  double updated_velocity = current_velocity_deg_per_s_ * (1 - update_rate) +
                            observed_velocity * update_rate;
  current_velocity_deg_per_s_ = updated_velocity > 0
                                    ? fmin(updated_velocity, max_velocity)
                                    : fmax(updated_velocity, -max_velocity);

  // Update prediction based on time input.
  return UpdatePrediction(time_us);
}

absl::Status KinematicPathSolver::UpdatePrediction(const int64_t time_us) {
  RET_CHECK(current_time_ < time_us)
      << "Prediction time added before a prior observation or prediction.";

  // Store prior pixel location.
  prior_position_px_ = current_position_px_;

  // Position update limited by min/max.
  double update_position_px =
      current_position_px_ +
      current_velocity_deg_per_s_ * mean_delta_t_ * pixels_per_degree_;

  if (update_position_px < min_location_) {
    current_position_px_ = min_location_;
    current_velocity_deg_per_s_ = 0;
    motion_state_ = false;
  } else if (update_position_px > max_location_) {
    current_position_px_ = max_location_;
    current_velocity_deg_per_s_ = 0;
    motion_state_ = false;
  } else {
    current_position_px_ = update_position_px;
  }
  current_time_ = time_us;

  return absl::OkStatus();
}

absl::Status KinematicPathSolver::GetState(int* position) {
  RET_CHECK(initialized_) << "GetState called before first observation added.";
  *position = round(current_position_px_);
  return absl::OkStatus();
}

absl::Status KinematicPathSolver::GetState(float* position) {
  RET_CHECK(initialized_) << "GetState called before first observation added.";
  *position = current_position_px_;
  return absl::OkStatus();
}

absl::Status KinematicPathSolver::GetDeltaState(float* delta_position) {
  RET_CHECK(initialized_) << "GetState called before first observation added.";
  *delta_position = current_position_px_ - prior_position_px_;
  return absl::OkStatus();
}

absl::Status KinematicPathSolver::SetState(const float position) {
  RET_CHECK(initialized_) << "SetState called before first observation added.";
  current_position_px_ = std::clamp(position, static_cast<float>(min_location_),
                                    static_cast<float>(max_location_));
  return absl::OkStatus();
}

absl::Status KinematicPathSolver::GetTargetPosition(int* target_position) {
  RET_CHECK(initialized_)
      << "GetTargetPosition called before first observation added.";

  // Provide target position clamped by min/max locations.
  if (target_position_px_ < min_location_) {
    *target_position = min_location_;
  } else if (target_position_px_ > max_location_) {
    *target_position = max_location_;
  } else {
    *target_position = round(target_position_px_);
  }
  return absl::OkStatus();
}

absl::Status KinematicPathSolver::UpdatePixelsPerDegree(
    const float pixels_per_degree) {
  RET_CHECK_GT(pixels_per_degree, 0)
      << "pixels_per_degree must be larger than 0.";
  pixels_per_degree_ = pixels_per_degree;
  return absl::OkStatus();
}

absl::Status KinematicPathSolver::UpdateMinMaxLocation(const int min_location,
                                                       const int max_location) {
  if (!initialized_) {
    max_location_ = max_location;
    min_location_ = min_location;
    return absl::OkStatus();
  }

  double prior_distance = max_location_ - min_location_;
  double updated_distance = max_location - min_location;
  double scale_change = updated_distance / prior_distance;
  current_position_px_ = current_position_px_ * scale_change;
  prior_position_px_ = prior_position_px_ * scale_change;
  target_position_px_ = target_position_px_ * scale_change;
  max_location_ = max_location;
  min_location_ = min_location;
  auto original_positions_at_time = raw_positions_at_time_;
  raw_positions_at_time_.clear();
  for (auto position_at_time : original_positions_at_time) {
    position_at_time.second = position_at_time.second * scale_change;
    raw_positions_at_time_.push_front(position_at_time);
  }
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
