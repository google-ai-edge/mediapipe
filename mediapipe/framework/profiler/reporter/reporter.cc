#include "mediapipe/framework/profiler/reporter/reporter.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "fstream"
#include "map"
#include "mediapipe/framework/calculator_profile.pb.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/re2.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace reporter {

const LazyRE2 kValidColumnRegex = {"^[a-zA-Z0-9_?*]+$"};
const LazyRE2 kReplace1WildcharRegex = {"\\?"};
const LazyRE2 kReplace0toNWildcharRegex = {"\\*"};

std::string ToStringF(double d) { return absl::StrFormat("%1.2f", d); }
std::string ToString(double d) { return absl::StrFormat("%1.0f", d); }

absl::btree_map<std::string,
                std::function<const std::string(const CalculatorData&)>>
    kColumns = {
        {"calculator",
         [](const CalculatorData& d) -> const std::string { return d.name; }},
        {"counter",
         [](const CalculatorData& d) -> const std::string {
           return ToString(d.counter);
         }},
        {"completed",
         [](const CalculatorData& d) -> const std::string {
           return ToString(d.completed);
         }},
        {"dropped",
         [](const CalculatorData& d) -> const std::string {
           return ToString(d.dropped);
         }},
        {"fps",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.fps);
         }},
        {"frequency",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.frequency);
         }},
        {"processing_rate",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.processing_rate);
         }},
        {"thread_count",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.thread_count);
         }},
        {"time_mean",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.time_stat.mean());
         }},
        {"time_stddev",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.time_stat.stddev());
         }},
        {"time_total",
         [](const CalculatorData& d) -> const std::string {
           return ToString(d.time_stat.total());
         }},
        {"time_percent",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.time_percent);
         }},
        {"input_latency_mean",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.input_latency_stat.mean());
         }},
        {"input_latency_stddev",
         [](const CalculatorData& d) -> const std::string {
           return ToStringF(d.input_latency_stat.stddev());
         }},
        {"input_latency_total",
         [](const CalculatorData& d) -> const std::string {
           return ToString(d.input_latency_stat.total());
         }}};

// Holds calculator traces that have an output trace with a provided stream ID
// and Packet timestamp.
typedef std::map<std::pair<int64_t, int32_t>,
                 const mediapipe::GraphTrace::CalculatorTrace*>
    PacketKeyToCalcTrace;

// Use this to locate the calculator trace with the start time for a given
// node its thread_id and its packet timestamp.
typedef std::map<std::pair<int64_t, std::pair<int32_t, int32_t>>,
                 const mediapipe::GraphTrace::CalculatorTrace*>
    TimestampNodeIdToCalcTrace;

// Maps node IDs to names.
typedef std::map<int32_t, std::string> NameLookup;

Reporter::Reporter() { MEDIAPIPE_CHECK_OK(set_columns({"*"})); }

int64_t RecursePacketStartTime(
    const PacketKeyToCalcTrace& output_trace_lookup,
    const mediapipe::GraphProfile& profile,
    const mediapipe::GraphTrace::CalculatorTrace& trace,
    std::vector<int32_t>* visited_calculators) {
  int64_t child_start_time =
      trace.has_start_time() ? trace.start_time() : trace.finish_time();
  const int32_t node_id = trace.node_id();
  if (std::find(visited_calculators->begin(), visited_calculators->end(),
                node_id) != visited_calculators->end()) {
    return child_start_time;
  }
  visited_calculators->push_back(node_id);

  for (size_t index = 0; index < trace.input_trace_size(); ++index) {
    const auto& stream_trace = trace.input_trace(index);

    // Find the output corresponding to this input.
    const auto it = output_trace_lookup.find(std::make_pair(
        stream_trace.packet_timestamp(), stream_trace.stream_id()));
    if (it != output_trace_lookup.end()) {
      child_start_time =
          std::min(child_start_time,
                   RecursePacketStartTime(output_trace_lookup, profile,
                                          *it->second, visited_calculators));
    }
  }
  return child_start_time;
}

int64_t CalculateInputLatency(
    const PacketKeyToCalcTrace& output_trace_lookup,
    const mediapipe::GraphProfile& profile,
    const mediapipe::GraphTrace::CalculatorTrace& trace) {
  // Track visited calculators to detect loops.
  std::vector<int> visited_calculators;

  // If a calculator has no start time, then there is no latency to measure.
  const auto result =
      !trace.has_start_time()
          ? 0L
          : trace.start_time() - RecursePacketStartTime(output_trace_lookup,
                                                        profile, trace,
                                                        &visited_calculators);

  return result;
}

void CacheNodeNameLookup(const mediapipe::GraphProfile& profile,
                         NameLookup* result) {
  for (const auto& graph_trace : profile.graph_trace()) {
    int key = 0;
    for (const auto& calc_name : graph_trace.calculator_name()) {
      (*result)[key] = calc_name;
      ++key;
    }
  }
}

void CacheOutputTraceLookup(const mediapipe::GraphProfile& profile,
                            PacketKeyToCalcTrace* output_trace_lookup,
                            TimestampNodeIdToCalcTrace* start_time_lookup) {
  for (const auto& graph_trace : profile.graph_trace()) {
    for (const auto& calc_trace : graph_trace.calculator_trace()) {
      if (calc_trace.event_type() != mediapipe::GraphTrace_EventType_PROCESS) {
        continue;
      }
      if (calc_trace.has_start_time() && !calc_trace.has_finish_time()) {
        (*start_time_lookup)[std::make_pair(
            calc_trace.input_timestamp(),
            std::make_pair(calc_trace.node_id(), calc_trace.thread_id()))] =
            &calc_trace;
      }
      for (const auto& stream_trace : calc_trace.output_trace()) {
        (*output_trace_lookup)[std::make_pair(stream_trace.packet_timestamp(),
                                              stream_trace.stream_id())] =
            &calc_trace;
      }
    }
  }
}

void CompleteCalculatorData(
    const GraphData& graph_data,
    std::map<std::string, CalculatorData>* calculator_data) {
  for (auto& calc_entry : *calculator_data) {
    auto& calc_data = calc_entry.second;
    const auto& time_stat = calc_data.time_stat;
    const auto& latency_stat = calc_data.input_latency_stat;
    const auto time_to_process = time_stat.mean() + latency_stat.mean();
    calc_data.fps = time_to_process == 0 ? 0 : 1.0E+6 / time_to_process;

    const auto duration = graph_data.max_time - graph_data.min_time;
    calc_data.frequency = calc_data.completed / (duration / 1.0E+6);
    calc_data.time_percent = 100 * time_stat.total() / graph_data.total_time;
    calc_data.dropped = calc_data.counter - calc_data.completed;

    calc_data.processing_rate = calc_data.time_stat.mean() == 0
                                    ? 0
                                    : 1.0 / calc_data.time_stat.mean() * 1.0E+6;
    calc_data.thread_count = calc_data.threads.size();
  }
}

void Reporter::Accumulate(const mediapipe::GraphProfile& profile) {
  // Cache nodeID to its std::string name.
  NameLookup name_lookup;
  CacheNodeNameLookup(profile, &name_lookup);

  // Cache some lookups so that we can quickly find the matching output stream
  // for a given input stream, and so that we can find the start time of a
  // given timestamp of a node.
  PacketKeyToCalcTrace output_trace_lookup;
  TimestampNodeIdToCalcTrace start_event_lookup;
  CacheOutputTraceLookup(profile, &output_trace_lookup, &start_event_lookup);

  // Hold the domain of all times found in the trace file.
  auto& min_time = graph_data_.min_time;
  auto& max_time = graph_data_.max_time;
  auto& total_time = graph_data_.total_time;

  // The start and finish time of PROCESS events might be split between
  // events. If a start event has been seen, we'll record it so that we can
  // match it up later.
  std::map<int32_t, absl::optional<int64_t>> start_times;

  for (const auto& graph_trace : profile.graph_trace()) {
    for (const auto& calc_trace : graph_trace.calculator_trace()) {
      if (calc_trace.event_type() != mediapipe::GraphTrace_EventType_PROCESS) {
        continue;
      }

      const auto& node_name = name_lookup[calc_trace.node_id()];
      auto& calc_data = calculator_data_[node_name];

      calc_data.name = node_name;
      calc_data.threads.insert(calc_trace.thread_id());

      // If there is a start time, update the domain of the trace time, and
      // mark that we've seen a start time for this calculator.
      if (calc_trace.has_start_time()) {
        min_time =
            std::min(min_time, static_cast<int64_t>(calc_trace.start_time() +
                                                    graph_trace.base_time()));
        ++calc_data.counter;
      }

      // If there is a finish time, update the domain and mark that an event
      // has been completed.
      if (calc_trace.has_finish_time()) {
        const auto finish_time =
            calc_trace.finish_time() + graph_trace.base_time();
        max_time = std::max(max_time, static_cast<int64_t>(finish_time));

        absl::optional<int64_t> start_time;
        if (calc_trace.has_start_time()) {
          start_time.emplace(calc_trace.start_time());
        } else {
          const auto start_event_it = start_event_lookup.find(std::make_pair(
              calc_trace.input_timestamp(),
              std::make_pair(calc_trace.node_id(), calc_trace.thread_id())));
          if (start_event_it != start_event_lookup.end()) {
            start_time.emplace(start_event_it->second->start_time());
          }
        }

        // Edge case -- If a finish time came in without a start time, then
        // we know that an event started before the trace became available.
        // But since we don't know when that is, we can't record its duration
        // and won't count it.
        if (start_time) {
          ++calc_data.completed;
          // Add up the duration of the events that led up to this start
          // event.
          const auto input_latency =
              CalculateInputLatency(output_trace_lookup, profile, calc_trace);
          calc_data.input_latency_stat.Push(input_latency);
          const auto duration =
              finish_time - (start_time.value() + graph_trace.base_time());
          calc_data.time_stat.Push(duration);
          total_time += duration;
        }
      }
    }
  }
}

::mediapipe::Status Reporter::set_columns(
    const std::vector<std::string>& columns) {
  bool error = false;
  std::stringstream warnings;
  std::vector<std::string> new_columns({"calculator"});

  // Iterate through the desired columns and build a regex.
  for (const auto& column_matcher : columns) {
    if (!RE2::PartialMatch(column_matcher, *kValidColumnRegex)) {
      warnings << "Column '" << column_matcher << "' is invalid." << std::endl;
      error = true;
      continue;
    }
    std::string colString = column_matcher;
    RE2::GlobalReplace(&colString, *kReplace0toNWildcharRegex, ".*");
    RE2::GlobalReplace(&colString, *kReplace1WildcharRegex, ".");

    // Iterator through our available columns and add them to our collection
    // of new columns if they do not already exist.
    bool matched = false;
    for (const auto& column : kColumns) {
      if (RE2::FullMatch(column.first, colString)) {
        matched = true;
        if (std::find(new_columns.begin(), new_columns.end(), column.first) ==
            new_columns.end()) {
          new_columns.push_back(column.first);
        }
      }
    }
    if (!matched) {
      warnings << "Column '" << column_matcher << "' did not match any columns."
               << std::endl;
      error = true;
    }
  }
  // As long as there is still one column, honor the request, even if error.
  if (!new_columns.empty()) {
    columns_.swap(new_columns);
  }
  if (!error) {
    return ::mediapipe::OkStatus();
  }
  return ::mediapipe::InvalidArgumentError(warnings.str());
}

class ReportImpl : public Report {
 public:
  ReportImpl(const std::map<std::string, CalculatorData>& calculator_data,
             const GraphData& graph_data)
      : calculator_data_(calculator_data), graph_data_(graph_data) {}
  void Print(std::ostream& output) override;
  const std::vector<std::string>& headers() override { return headers_impl; }
  const std::vector<std::vector<std::string>>& lines() override {
    return lines_impl;
  }
  const GraphData& graph_data() override { return graph_data_; }
  const std::map<std::string, CalculatorData>& calculator_data() override {
    return calculator_data_;
  }

  // Each header name in alphabetical order, except the first column, which is
  // always "calculator".
  std::vector<std::string> headers_impl;

  // Values for each calculator, corresponding to the label in headers().
  std::vector<std::vector<std::string>> lines_impl;

  // The longest std::string of any value in a given column (including the
  // header for that column). Used for formatting the output.
  std::vector<size_t> char_counts_impl;
  bool compact_flag = false;

  const std::map<std::string, CalculatorData>& calculator_data_;
  const GraphData& graph_data_;
};

void ReportImpl::Print(std::ostream& output) {
  // Print the results, but aside from the last column, add whitespace to
  // fill space up to char_counts[column] + 1. The strings in the output
  // are mutable to support padding, hence no const in the for loops.
  int column_number = 0;
  // Make a copy of the column std::string because we might be adding spaces.
  for (auto column : headers_impl) {
    int padding_needed = char_counts_impl[column_number] + 1 - column.length();
    if (compact_flag) {
      padding_needed = 1;
    }
    column.append(padding_needed, ' ');
    output << column;
    column_number++;
  }
  output << std::endl;

  for (auto& row : lines_impl) {
    int column_number = 0;
    for (auto column : row) {
      int padding_needed =
          char_counts_impl[column_number] + 1 - column.length();
      if (compact_flag) {
        padding_needed = 1;
      }
      column.append(padding_needed, ' ');
      output << column;
      column_number++;
    }
    output << std::endl;
  }
}

std::unique_ptr<Report> Reporter::Report() {
  CompleteCalculatorData(graph_data_, &calculator_data_);

  auto report = std::make_unique<ReportImpl>(calculator_data_, graph_data_);
  report->compact_flag = compact_flag_;

  // First row contains the column headers.
  auto& headers = report->headers_impl;
  auto& lines = report->lines_impl;
  auto& char_counts = report->char_counts_impl;

  headers = columns_;
  char_counts.resize(headers.size());
  std::transform(headers.begin(), headers.end(), char_counts.begin(),
                 [](const auto& header) { return header.size(); });

  for (const auto& header : headers) {
    size_t line_num = 0;
    for (auto calc_it = calculator_data_.begin();
         calc_it != calculator_data_.end(); ++calc_it) {
      const std::string value = kColumns[header](calc_it->second);
      if (calc_it->second.name.empty()) {
        continue;
      }
      while (line_num >= lines.size()) {
        lines.push_back({});
      }
      auto& line = lines[line_num];
      line.push_back(value);
      const size_t char_count_index = line.size() - 1;
      char_counts[char_count_index] =
          std::max(char_counts[char_count_index], value.length());
      ++line_num;
    }
  }
  return report;
}

}  // namespace reporter
}  // namespace mediapipe
