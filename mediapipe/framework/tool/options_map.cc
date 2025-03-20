#include "mediapipe/framework/tool/options_map.h"

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"

namespace mediapipe {
namespace tool {

ABSL_CONST_INIT absl::Mutex option_extension_lock(absl::kConstInit);

}  // namespace tool
}  // namespace mediapipe
