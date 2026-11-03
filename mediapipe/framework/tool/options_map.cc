#include "mediapipe/framework/tool/options_map.h"

#include "absl/base/attributes.h"        // from @com_google_absl
#include "absl/base/const_init.h"        // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace mediapipe {
namespace tool {

ABSL_CONST_INIT absl::Mutex option_extension_lock(absl::kConstInit);

}  // namespace tool
}  // namespace mediapipe
