#include "mediapipe/framework/tool/graph_runtime_info_utils.h"

#include <algorithm>
#include <set>
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
  std::set<std::string> calculators_with_unprocessed_packets;
  std::vector<std::string> running_calculators;

  // Analyze packets in input queues.
  int num_total_pending_packets = 0;
  for (const auto& calculator_info : graph_runtime_info.calculator_infos()) {
    const bool is_idle = calculator_info.last_process_finish_unix_us() >=
                         calculator_info.last_process_start_unix_us();
    int calculator_pending_packets = 0;
    Timestamp min_ts_bound_of_streams_with_unprocessed_packets =
        Timestamp::Max();
    for (const auto& input_stream_info : calculator_info.input_stream_infos()) {
      calculator_pending_packets += input_stream_info.queue_size();
      num_total_pending_packets += input_stream_info.queue_size();
      const Timestamp stream_ts_bound = Timestamp::CreateNoErrorChecking(
          input_stream_info.minimum_timestamp_or_bound());
      if (input_stream_info.queue_size() > 0) {
        min_ts_bound_of_streams_with_unprocessed_packets = std::min(
            min_ts_bound_of_streams_with_unprocessed_packets, stream_ts_bound);
      }
    }
    // Determine calculator state.
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
    const Timestamp calculator_ts_bound =
        Timestamp::CreateNoErrorChecking(calculator_info.timestamp_bound());
    absl::StrAppend(
        &calculators_runtime_info_str,
        absl::StrFormat("\n%s: (%s%s, ts bound : %s)\n",
                        calculator_info.calculator_name(), calculator_state_str,
                        calculator_pending_packets > 0
                            ? absl::StrCat(", pending packets: ",
                                           calculator_pending_packets)
                            : "",
                        calculator_ts_bound.DebugString()));
    if (calculator_pending_packets > 0) {
      // Predict streams that might be waiting for packets.
      std::vector<std::string> streams_with_waiting_for_packets;
      for (const auto& input_stream_info :
           calculator_info.input_stream_infos()) {
        const Timestamp stream_ts_bound = Timestamp::CreateNoErrorChecking(
            input_stream_info.minimum_timestamp_or_bound());
        if (stream_ts_bound <
            min_ts_bound_of_streams_with_unprocessed_packets) {
          streams_with_waiting_for_packets.push_back(
              input_stream_info.stream_name());
        }
      }
      const std::string waiting_for_packets_str =
          absl::StrCat("waiting on stream(s): ",
                       absl::StrJoin(streams_with_waiting_for_packets, ", "));
      absl::StrAppend(&calculators_runtime_info_str, waiting_for_packets_str,
                      "\n");
      calculators_with_unprocessed_packets.insert(absl::StrCat(
          calculator_info.calculator_name(), " ", waiting_for_packets_str));
    }

    // List input streams with state.
    if (!calculator_info.input_stream_infos().empty()) {
      absl::StrAppend(&calculators_runtime_info_str, "Input streams:\n");
    }
    for (const auto& input_stream_info : calculator_info.input_stream_infos()) {
      absl::StrAppend(
          &calculators_runtime_info_str, " * ", input_stream_info.stream_name(),
          " - queue size: ", input_stream_info.queue_size(),
          ", total added: ", input_stream_info.number_of_packets_added(),
          ", ts bound: ",
          Timestamp::CreateNoErrorChecking(
              input_stream_info.minimum_timestamp_or_bound())
              .DebugString(),
          "\n");
    }
    // List output streams with state.
    if (!calculator_info.output_stream_infos().empty()) {
      absl::StrAppend(&calculators_runtime_info_str, "Output streams:\n");
    }
    for (const auto& output_stream_info :
         calculator_info.output_stream_infos()) {
      absl::StrAppend(&calculators_runtime_info_str, " * ",
                      output_stream_info.stream_name(), ", total added: ",
                      output_stream_info.number_of_packets_added(),
                      ", ts bound: ",
                      Timestamp::CreateNoErrorChecking(
                          output_stream_info.minimum_timestamp_or_bound())
                          .DebugString(),
                      "\n");
    }
  }

  const std::string calulators_with_unprocessed_packets_str =
      !calculators_with_unprocessed_packets.empty()
          ? absl::StrCat(
                absl::StrJoin(calculators_with_unprocessed_packets, "\n"))
          : "\n";
  const std::string running_calculators_str =
      running_calculators.empty()
          ? "None"
          : absl::StrCat(" (running calculators: ",
                         absl::StrJoin(running_calculators, ", "), ")");
  return absl::StrFormat(
      "Graph runtime info: \nRunning calculators: %s\nNum packets in input "
      "queues: %d\n%s\n%s\n",
      running_calculators_str, num_total_pending_packets,
      calulators_with_unprocessed_packets_str, calculators_runtime_info_str);
}

}  // namespace mediapipe::tool
