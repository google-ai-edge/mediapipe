#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

StatusOr<std::string> GetDefaultTraceLogDirectory() {
  return "/data/local/tmp";
}

}  // namespace mediapipe
