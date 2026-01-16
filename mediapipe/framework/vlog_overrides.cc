#include "mediapipe/framework/vlog_overrides.h"

// Template to temporary enable VLOG overrides in code:
// #define MEDIAPIPE_VLOG_VMODULE "calculator_graph*=5,southbound*=5"
// #define MEDIAPIPE_VLOG_V 1

#if defined(MEDIAPIPE_VLOG_V) || defined(MEDIAPIPE_VLOG_VMODULE)

#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/globals.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/no_destructor.h"

#endif  // defined(MEDIAPIPE_VLOG_V) || defined(MEDIAPIPE_VLOG_VMODULE)

namespace mediapipe {

void SetVLogOverrides() {
#if defined(MEDIAPIPE_VLOG_V)
  ABSL_LOG(INFO) << absl::StrFormat("Setting global VLOG level: %d",
                                    MEDIAPIPE_VLOG_V);
  absl::SetGlobalVLogLevel(MEDIAPIPE_VLOG_V);
#endif  // defined(MEDIAPIPE_VLOG_V)

#if defined(MEDIAPIPE_VLOG_VMODULE)
  static NoDestructor<std::vector<std::pair<std::string, int>>> kVModuleMapping(
      []() {
        constexpr absl::string_view kVModule = MEDIAPIPE_VLOG_VMODULE;
        std::vector<std::string> parts =
            absl::StrSplit(kVModule, absl::ByAnyChar(",="));
        ABSL_CHECK_EQ(parts.size() % 2, 0)
            << "Invalid MEDIAPIPE_VLOG_VMODULE: " << kVModule;
        std::vector<std::pair<std::string, int>> result;
        for (int i = 0; i < parts.size(); i += 2) {
          result.push_back({parts[i], std::stoi(parts[i + 1])});
        }
        return result;
      }());

  ABSL_LOG(INFO) << "Setting VLOG levels...";
  for (const auto& [key, value] : *kVModuleMapping) {
    ABSL_LOG(INFO) << absl::StrFormat("Setting [%s] to level: %d", key, value);
    absl::SetVLogLevel(key, value);
  }
#endif  // defined(MEDIAPIPE_VLOG_VMODULE)
}

}  // namespace mediapipe
