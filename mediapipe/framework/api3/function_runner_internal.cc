#include "mediapipe/framework/api3/function_runner_internal.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/output_stream_poller.h"

namespace mediapipe::api3 {

absl::StatusOr<mediapipe::Packet> GetOutputPacket(
    OutputStreamPoller& poller, const ErrorCallback& error_callback) {
  mediapipe::Packet packet;
  if (!poller.Next(&packet)) {
    if (error_callback.HasErrors()) {
      std::vector<absl::Status> errors = error_callback.GetErrors();
      if (errors.size() == 1) return std::move(errors[0]);
      return tool::CombinedStatus("Failed to poll the output", errors);
    } else {
      return absl::InternalError("Failled to poll the output.");
    }
  }

  // NOTE: currently supporting timestamp-less execution only.
  return std::move(packet).At(Timestamp::Unset());
}

}  // namespace mediapipe::api3
