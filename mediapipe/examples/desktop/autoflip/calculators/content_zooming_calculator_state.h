#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_CALCULATORS_CONTENT_ZOOMING_CALCULATOR_STATE_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_CALCULATORS_CONTENT_ZOOMING_CALCULATOR_STATE_H_

#include <optional>

#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace autoflip {

struct ContentZoomingCalculatorState {
  int frame_height = -1;
  int frame_width = -1;
  // Path solver used to smooth top/bottom border crop values.
  KinematicPathSolver path_solver_zoom;
  KinematicPathSolver path_solver_pan;
  KinematicPathSolver path_solver_tilt;
  // Stores the time of the first crop rectangle.
  Timestamp first_rect_timestamp;
  // Stores the first crop rectangle.
  mediapipe::NormalizedRect first_rect;
  // Stores the time of the last "only_required" input.
  int64 last_only_required_detection = 0;
  // Rect values of last message with detection(s).
  int last_measured_height = 0;
  int last_measured_x_offset = 0;
  int last_measured_y_offset = 0;
};

using ContentZoomingCalculatorStateCacheType =
    std::optional<ContentZoomingCalculatorState>;

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_CALCULATORS_CONTENT_ZOOMING_CALCULATOR_STATE_H_
