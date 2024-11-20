#include "mediapipe/framework/tool/graph_runtime_info_utils.h"

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe::tool {

absl::StatusOr<std::string> GetGraphRuntimeInfoString(
    const GraphRuntimeInfo& graph_runtime_info) {
  const absl::Time caputure_time =
      absl::FromUnixMicros(graph_runtime_info.capture_time_unix_us());
  std::string calculators_runtime_info_str;
  std::vector<std::string> calculators_with_unprocessed_packets;
  std::vector<std::string> running_calculators;
  int num_packets_in_input_queues = 0;
  for (const auto& calculator_info : graph_runtime_info.calculator_infos()) {
    const bool is_idle = calculator_info.last_process_finish_unix_us() >=
                         calculator_info.last_process_start_unix_us();
    const std::string calculator_state_str =
        is_idle ? absl::StrFormat(
                      "idle for %.2fs",
                      absl::ToDoubleSeconds(
                          caputure_time -
                          absl::FromUnixMicros(
                              calculator_info.last_process_finish_unix_us())))
                : absl::StrFormat(
                      "running for %.2fs",
                      absl::ToDoubleSeconds(
                          caputure_time -
                          absl::FromUnixMicros(
                              calculator_info.last_process_start_unix_us())));
    if (!is_idle) {
      running_calculators.push_back(calculator_info.calculator_name());
    }
    absl::StrAppend(
        &calculators_runtime_info_str,
        absl::StrFormat(
            "\n%s: (%s, ts bound : %s)", calculator_info.calculator_name(),
            calculator_state_str,
            Timestamp::CreateNoErrorChecking(calculator_info.timestamp_bound())
                .DebugString()));
    bool calculator_has_unprocessed_packets = false;
    for (const auto& input_stream_info : calculator_info.input_stream_infos()) {
      num_packets_in_input_queues += input_stream_info.queue_size();
      calculator_has_unprocessed_packets |= input_stream_info.queue_size() > 0;
      absl::StrAppend(
          &calculators_runtime_info_str, " * ", input_stream_info.stream_name(),
          " - queue size: ", input_stream_info.queue_size(),
          ", total added: ", input_stream_info.number_of_packets_added(),
          ", min ts: ",
          Timestamp::CreateNoErrorChecking(
              input_stream_info.minimum_timestamp_or_bound())
              .DebugString(),
          "\n");
    }
    if (calculator_has_unprocessed_packets) {
      calculators_with_unprocessed_packets.push_back(
          calculator_info.calculator_name());
    }
  }
  const std::string calulators_with_unprocessed_packets_str =
      num_packets_in_input_queues > 0
          ? absl::StrCat(
                " (in calculators: ",
                absl::StrJoin(calculators_with_unprocessed_packets, ", "), ")")
          : "";
  const std::string running_calculators_str =
      running_calculators.empty()
          ? "None"
          : absl::StrCat(" (running calculators: ",
                         absl::StrJoin(running_calculators, ", "), ")");
  return absl::StrFormat(
      "Graph runtime info: \nRunning calculators: %s\nNum packets in input "
      "queues: %d%s\n%s\n",
      running_calculators_str, num_packets_in_input_queues,
      calulators_with_unprocessed_packets_str, calculators_runtime_info_str);
}

}  // namespace mediapipe::tool
