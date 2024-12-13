#include "mediapipe/framework/tool/graph_runtime_info_logger.h"

#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/tool/graph_runtime_info_utils.h"
#include "mediapipe/framework/vlog_utils.h"

namespace mediapipe::tool {

constexpr absl::Duration kDefaultCaptureInterval = absl::Seconds(10);

GraphRuntimeInfoLogger::GraphRuntimeInfoLogger()
    : thread_pool_("GraphRuntimeInfoLogger", /*num_threads=*/1) {}

GraphRuntimeInfoLogger::~GraphRuntimeInfoLogger() { Stop(); };

absl::Status GraphRuntimeInfoLogger::StartInBackground(
    const mediapipe::GraphRuntimeInfoConfig& config,
    absl::AnyInvocable<absl::StatusOr<GraphRuntimeInfo>()>
        get_runtime_info_fn) {
  get_runtime_info_fn_ = std::move(get_runtime_info_fn);
  RET_CHECK(!is_running_.HasBeenNotified());
  ABSL_CHECK_EQ(thread_pool_.num_threads(), 1);
  thread_pool_.StartWorkers();
  absl::Duration interval =
      config.capture_period_msec() > 0
          ? absl::Milliseconds(config.capture_period_msec())
          : kDefaultCaptureInterval;
  thread_pool_.Schedule([this, interval]() mutable {
    is_running_.Notify();
    while (!shutdown_signal_.HasBeenNotified()) {
      const auto runtime_info = get_runtime_info_fn_();
      if (!runtime_info.ok()) {
        ABSL_LOG(DFATAL) << "Failed to get graph runtime info: "
                         << runtime_info.status();
        return;
      }
      const auto runtime_info_str = GetGraphRuntimeInfoString(*runtime_info);
      if (!runtime_info_str.ok()) {
        ABSL_LOG(DFATAL) << "Failed to render graph runtime info: "
                         << runtime_info_str.status();
        return;
      }
      VlogLargeMessage(/*verbose_level=*/0, *runtime_info_str);
      shutdown_signal_.WaitForNotificationWithTimeout(interval);
    }
  });
  is_running_.WaitForNotification();
  return absl::OkStatus();
}

void GraphRuntimeInfoLogger::Stop() { shutdown_signal_.Notify(); }

}  // namespace mediapipe::tool
