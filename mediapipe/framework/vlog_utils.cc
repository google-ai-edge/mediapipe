#include "mediapipe/framework/vlog_utils.h"

#include "absl/log/absl_log.h"       // from @com_google_absl
#include "absl/log/log.h"            // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

void VlogLargeMessage(int verbose_level, absl::string_view message) {
#if defined(MEDIAPIPE_MOBILE)
  if (message.size() > 4096) {
    for (const auto& line : absl::StrSplit(message, '\n')) {
      VLOG(verbose_level) << line;
    }
    return;
  }
#endif
  VLOG(verbose_level) << message;
}

}  // namespace mediapipe
