// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/profiler/graph_profiler.h"

#include <fstream>
#include <list>

#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/advanced_proto_lite_inc.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/re2.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/profiler/profiler_resource_util.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/validate_name.h"

namespace mediapipe {

using tool::TagMap;

namespace {

const int kDefaultLogIntervalCount = 10;
const int kDefaultLogFileCount = 2;
const char kDefaultLogFilePrefix[] = "mediapipe_trace_";

// The number of recent timestamps tracked for each input stream.
const int kPacketInfoRecentCount = 400;

std::string PacketIdToString(const PacketId& packet_id) {
  return absl::Substitute("stream_name: $0, timestamp_usec: $1",
                          packet_id.stream_name, packet_id.timestamp_usec);
}

int GetLogIntervalCount(const ProfilerConfig& profiler_config) {
  return profiler_config.trace_log_interval_count()
             ? profiler_config.trace_log_interval_count()
             : kDefaultLogIntervalCount;
}

int GetLogFileCount(const ProfilerConfig& profiler_config) {
  return profiler_config.trace_log_count() ? profiler_config.trace_log_count()
                                           : kDefaultLogFileCount;
}

// Returns true if aggregate timing data is recorded.
bool IsProfilerEnabled(const ProfilerConfig& profiler_config) {
  return profiler_config.enable_profiler();
}

// Returns true if trace events are recorded.
bool IsTracerEnabled(const ProfilerConfig& profiler_config) {
  return profiler_config.trace_enabled();
}

// Returns true if trace events are written to a log file.
// Note that for now, file output is only for graph-trace and not for
// calculator-profile.
bool IsTraceLogEnabled(const ProfilerConfig& profiler_config) {
  return IsTracerEnabled(profiler_config) &&
         !profiler_config.trace_log_disabled();
}

// Returns true if trace events are written periodically.
bool IsTraceIntervalEnabled(const ProfilerConfig& profiler_config,
                            GraphTracer* tracer) {
  return IsTraceLogEnabled(profiler_config) && tracer &&
         absl::ToInt64Microseconds(tracer->GetTraceLogInterval()) != -1;
}

using PacketInfoMap =
    ShardedMap<std::string, std::list<std::pair<int64, PacketInfo>>>;

// Inserts a PacketInfo into a PacketInfoMap.
void InsertPacketInfo(PacketInfoMap* map, const PacketId& packet_id,
                      const PacketInfo& packet_info) {
  auto entry = map->find(packet_id.stream_name);
  if (entry == map->end()) {
    entry = map->insert({packet_id.stream_name, {}}).first;
  }
  auto& list = entry->second;
  list.push_back({packet_id.timestamp_usec, packet_info});
  while (list.size() > kPacketInfoRecentCount) {
    list.pop_front();
  }
}

// Returns a PacketInfo from a PacketInfoMap.
PacketInfo* GetPacketInfo(PacketInfoMap* map, const PacketId& packet_id) {
  auto entry = map->find(packet_id.stream_name);
  if (entry == map->end()) {
    return nullptr;
  }
  auto& list = entry->second;
  for (auto iter = list.rbegin(); iter != list.rend(); ++iter) {
    if (iter->first == packet_id.timestamp_usec) {
      return &iter->second;
    }
  }
  return nullptr;
}

}  // namespace

// Builds GraphProfile records from profiler timing data.
class GraphProfiler::GraphProfileBuilder {
 public:
  GraphProfileBuilder(GraphProfiler* profiler)
      : profiler_(profiler), calculator_regex_(".*") {
    auto& filter = profiler_->profiler_config().calculator_filter();
    calculator_regex_ = filter.empty() ? calculator_regex_ : RE2(filter);
  }

  bool ProfileIncluded(const CalculatorProfile& p) {
    return RE2::FullMatch(p.name(), calculator_regex_);
  }

 private:
  GraphProfiler* profiler_;
  RE2 calculator_regex_;
};

GraphProfiler::GraphProfiler()
    : is_initialized_(false),
      is_profiling_(false),
      calculator_profiles_(1000),
      packets_info_(1000),
      is_running_(false),
      previous_log_end_time_(absl::InfinitePast()),
      previous_log_index_(-1),
      validated_graph_(nullptr) {
  clock_ = std::shared_ptr<mediapipe::Clock>(
      mediapipe::MonotonicClock::CreateSynchronizedMonotonicClock());
}

GraphProfiler::~GraphProfiler() {}

void GraphProfiler::Initialize(
    const ValidatedGraphConfig& validated_graph_config) {
  absl::WriterMutexLock lock(&profiler_mutex_);
  validated_graph_ = &validated_graph_config;
  CHECK(!is_initialized_)
      << "Cannot initialize the profiler for the same graph multiple times.";
  profiler_config_ = validated_graph_config.Config().profiler_config();
  int64 interval_size_usec = profiler_config_.histogram_interval_size_usec();
  interval_size_usec = interval_size_usec ? interval_size_usec : 1000000;
  int64 num_intervals = profiler_config_.num_histogram_intervals();
  num_intervals = num_intervals ? num_intervals : 1;
  if (IsTracerEnabled(profiler_config_)) {
    packet_tracer_ = absl::make_unique<GraphTracer>(profiler_config_);
  }
  for (int node_id = 0;
       node_id < validated_graph_config.CalculatorInfos().size(); ++node_id) {
    std::string node_name =
        tool::CanonicalNodeName(validated_graph_config.Config(), node_id);
    CalculatorProfile profile;
    profile.set_name(node_name);
    InitializeTimeHistogram(interval_size_usec, num_intervals,
                            profile.mutable_process_runtime());
    if (profiler_config_.enable_stream_latency()) {
      InitializeTimeHistogram(interval_size_usec, num_intervals,
                              profile.mutable_process_input_latency());
      InitializeTimeHistogram(interval_size_usec, num_intervals,
                              profile.mutable_process_output_latency());

      const CalculatorGraphConfig::Node& node_config =
          validated_graph_config.Config().node(node_id);
      InitializeOutputStreams(node_config);
      InitializeInputStreams(node_config, interval_size_usec, num_intervals,
                             &profile);
    }

    auto iter = calculator_profiles_.insert({node_name, profile});
    CHECK(iter.second) << absl::Substitute(
        "Calculator \"$0\" has already been added.", node_name);
  }
  profile_builder_ = std::make_unique<GraphProfileBuilder>(this);

  is_initialized_ = true;
}

void GraphProfiler::SetClock(const std::shared_ptr<mediapipe::Clock>& clock) {
  absl::WriterMutexLock lock(&profiler_mutex_);
  CHECK(clock) << "GraphProfiler::SetClock() is called with a nullptr.";
  clock_ = clock;
}

const std::shared_ptr<mediapipe::Clock> GraphProfiler::GetClock() const {
  return clock_;
}

void GraphProfiler::Pause() {
  is_profiling_ = false;
  is_tracing_ = false;
}

void GraphProfiler::Resume() {
  // is_profiling_ enables recording of performance stats.
  // is_tracing_ enables recording of timing events.
  // While the graph is running, these variables indicate
  // IsProfilerEnabled and IsTracerEnabled.
  is_profiling_ = IsProfilerEnabled(profiler_config_);
  is_tracing_ = IsTracerEnabled(profiler_config_);
}

void GraphProfiler::Reset() {
  absl::WriterMutexLock lock(&profiler_mutex_);
  for (auto iter = calculator_profiles_.begin();
       iter != calculator_profiles_.end(); ++iter) {
    CalculatorProfile* calculator_profile = &iter->second;
    ResetTimeHistogram(calculator_profile->mutable_process_runtime());
    ResetTimeHistogram(calculator_profile->mutable_process_input_latency());
    ResetTimeHistogram(calculator_profile->mutable_process_output_latency());
    for (auto& input_stream_profile :
         *(calculator_profile->mutable_input_stream_profiles())) {
      ResetTimeHistogram(input_stream_profile.mutable_latency());
    }
  }
}

// Begins profiling for a single graph run.
absl::Status GraphProfiler::Start(mediapipe::Executor* executor) {
  // If specified, start periodic profile output while the graph runs.
  Resume();
  if (is_tracing_ && IsTraceIntervalEnabled(profiler_config_, tracer()) &&
      executor != nullptr) {
    // Inform the user via logging the path to the trace logs.
    ASSIGN_OR_RETURN(std::string trace_log_path, GetTraceLogPath());
    LOG(INFO) << "trace_log_path: " << trace_log_path;

    is_running_ = true;
    executor->Schedule([this] {
      absl::Time deadline = clock_->TimeNow() + tracer()->GetTraceLogInterval();
      while (is_running_) {
        clock_->SleepUntil(deadline);
        deadline = clock_->TimeNow() + tracer()->GetTraceLogInterval();
        if (is_running_) {
          WriteProfile().IgnoreError();
        }
      }
    });
  }
  return absl::OkStatus();
}

// Ends profiling for a single graph run.
absl::Status GraphProfiler::Stop() {
  is_running_ = false;
  Pause();
  // If specified, write a final profile.
  if (IsTraceLogEnabled(profiler_config_)) {
    MP_RETURN_IF_ERROR(WriteProfile());
  }
  return absl::OkStatus();
}

void GraphProfiler::LogEvent(const TraceEvent& event) {
  // Record event info in the event trace log.

  if (packet_tracer_) {
    if (event.event_type == GraphTrace::GPU_TASK ||
        event.event_type == GraphTrace::GPU_CALIBRATION) {
      packet_tracer_->LogEvent(event);
    } else {
      absl::Time time_now = clock_->TimeNow();
      packet_tracer_->LogEvent(TraceEvent(event).set_event_time(time_now));
    }
  }

  // Record event info in the profiling histograms.
  if (event.event_type == GraphTrace::PROCESS && event.node_id == -1) {
    AddPacketInfo(event);
  }
}

void GraphProfiler::AddPacketInfo(const TraceEvent& packet_info) {
  absl::ReaderMutexLock lock(&profiler_mutex_);
  if (!is_profiling_) {
    return;
  }

  Timestamp packet_timestamp = packet_info.input_ts;
  std::string stream_name = *packet_info.stream_id;

  if (!profiler_config_.enable_stream_latency()) {
    return;
  }
  if (!packet_timestamp.IsRangeValue()) {
    LOG(WARNING) << absl::Substitute(
        "Skipped adding packet info because the timestamp $0 for stream "
        "\"$1\" is not valid.",
        packet_timestamp.Value(), stream_name);
    return;
  }

  int64 production_time_usec =
      profiler_config_.use_packet_timestamp_for_added_packet()
          ? packet_timestamp.Value()
          : TimeNowUsec();
  AddPacketInfoInternal(PacketId({stream_name, packet_timestamp.Value()}),
                        production_time_usec, production_time_usec);
}

absl::Status GraphProfiler::GetCalculatorProfiles(
    std::vector<CalculatorProfile>* profiles) const {
  absl::ReaderMutexLock lock(&profiler_mutex_);
  RET_CHECK(is_initialized_)
      << "GetCalculatorProfiles can only be called after Initialize()";
  for (auto& entry : calculator_profiles_) {
    profiles->push_back(entry.second);
  }
  return absl::OkStatus();
}

void GraphProfiler::InitializeTimeHistogram(int64 interval_size_usec,
                                            int64 num_intervals,
                                            TimeHistogram* histogram) {
  histogram->set_interval_size_usec(interval_size_usec);
  histogram->set_num_intervals(num_intervals);
  histogram->mutable_count()->Resize(num_intervals, /*value=*/0);
  ResetTimeHistogram(histogram);
}

void GraphProfiler::InitializeOutputStreams(
    const CalculatorGraphConfig::Node& node_config) {}

void GraphProfiler::InitializeInputStreams(
    const CalculatorGraphConfig::Node& node_config, int64 interval_size_usec,
    int64 num_intervals, CalculatorProfile* calculator_profile) {
  std::shared_ptr<tool::TagMap> input_tag_map =
      TagMap::Create(node_config.input_stream()).value();
  std::set<int> back_edge_ids = GetBackEdgeIds(node_config, *input_tag_map);
  auto input_tag_map_names = input_tag_map->Names();
  for (int i = 0; i < input_tag_map_names.size(); ++i) {
    std::string input_stream_name = input_tag_map_names[i];
    StreamProfile* input_stream_profile =
        calculator_profile->add_input_stream_profiles();
    input_stream_profile->set_name(input_stream_name);
    input_stream_profile->set_back_edge(back_edge_ids.find(i) !=
                                        back_edge_ids.end());
    InitializeTimeHistogram(interval_size_usec, num_intervals,
                            input_stream_profile->mutable_latency());
  }
}

std::set<int> GraphProfiler::GetBackEdgeIds(
    const CalculatorGraphConfig::Node& node_config,
    const TagMap& input_tag_map) {
  std::set<int> back_edge_ids;
  for (const auto& input_stream_info : node_config.input_stream_info()) {
    if (!input_stream_info.back_edge()) {
      continue;
    }
    std::string tag;
    int index;
    MEDIAPIPE_CHECK_OK(
        tool::ParseTagIndex(input_stream_info.tag_index(), &tag, &index))
        << absl::Substitute("Cannot parse TAG or index for the backedge \"$0\"",
                            input_stream_info.tag_index());
    CHECK(0 <= index && index < input_tag_map.NumEntries(tag))
        << absl::Substitute(
               "The input_stream_info for tag \"$0\" (index "
               "$1) does not match any input_stream.",
               tag, index);
    back_edge_ids.insert(input_tag_map.GetId(tag, index).value());
  }
  return back_edge_ids;
}

void GraphProfiler::ResetTimeHistogram(TimeHistogram* histogram) {
  histogram->set_total(0);
  for (auto& count : *(histogram->mutable_count())) {
    count = 0;
  }
}

void GraphProfiler::AddPacketInfoInternal(const PacketId& packet_id,
                                          int64 production_time_usec,
                                          int64 source_process_start_usec) {
  PacketInfo packet_info = {0, production_time_usec, source_process_start_usec};
  InsertPacketInfo(&packets_info_, packet_id, packet_info);
}

void GraphProfiler::AddPacketInfoForOutputPackets(
    const OutputStreamShardSet& output_stream_shard_set,
    int64 production_time_usec, int64 source_process_start_usec) {
  for (const OutputStreamShard& output_stream_shard : output_stream_shard_set) {
    for (const Packet& output_packet : *output_stream_shard.OutputQueue()) {
      AddPacketInfoInternal(PacketId({output_stream_shard.Name(),
                                      output_packet.Timestamp().Value()}),
                            production_time_usec, source_process_start_usec);
    }
  }
}

int64 GraphProfiler::AddStreamLatencies(
    const CalculatorContext& calculator_context, int64 start_time_usec,
    int64 end_time_usec, CalculatorProfile* calculator_profile) {
  // Update input streams profiles.
  int64 min_source_process_start_usec = AddInputStreamTimeSamples(
      calculator_context, start_time_usec, calculator_profile);

  // Update output production times.
  AddPacketInfoForOutputPackets(calculator_context.Outputs(), end_time_usec,
                                min_source_process_start_usec);
  return min_source_process_start_usec;
}

void GraphProfiler::SetOpenRuntime(const CalculatorContext& calculator_context,
                                   int64 start_time_usec, int64 end_time_usec) {
  absl::ReaderMutexLock lock(&profiler_mutex_);
  if (!is_profiling_) {
    return;
  }

  const std::string& node_name = calculator_context.NodeName();
  int64 time_usec = end_time_usec - start_time_usec;
  auto profile_iter = calculator_profiles_.find(node_name);
  CHECK(profile_iter != calculator_profiles_.end()) << absl::Substitute(
      "Calculator \"$0\" has not been added during initialization.",
      calculator_context.NodeName());
  CalculatorProfile* calculator_profile = &profile_iter->second;
  calculator_profile->set_open_runtime(time_usec);

  if (profiler_config_.enable_stream_latency()) {
    AddStreamLatencies(calculator_context, start_time_usec, end_time_usec,
                       calculator_profile);
  }
}

void GraphProfiler::SetCloseRuntime(const CalculatorContext& calculator_context,
                                    int64 start_time_usec,
                                    int64 end_time_usec) {
  absl::ReaderMutexLock lock(&profiler_mutex_);
  if (!is_profiling_) {
    return;
  }
  const std::string& node_name = calculator_context.NodeName();
  int64 time_usec = end_time_usec - start_time_usec;
  auto profile_iter = calculator_profiles_.find(node_name);
  CHECK(profile_iter != calculator_profiles_.end()) << absl::Substitute(
      "Calculator \"$0\" has not been added during initialization.",
      calculator_context.NodeName());
  CalculatorProfile* calculator_profile = &profile_iter->second;
  calculator_profile->set_close_runtime(time_usec);

  if (profiler_config_.enable_stream_latency()) {
    AddStreamLatencies(calculator_context, start_time_usec, end_time_usec,
                       calculator_profile);
  }
}

void GraphProfiler::AddTimeSample(int64 start_time_usec, int64 end_time_usec,
                                  TimeHistogram* histogram) {
  if (end_time_usec < start_time_usec) {
    LOG(ERROR) << absl::Substitute(
        "end_time_usec ($0) is < start_time_usec ($1)", end_time_usec,
        start_time_usec);
    return;
  }

  int64 time_usec = end_time_usec - start_time_usec;
  histogram->set_total(histogram->total() + time_usec);
  int64 interval_index = time_usec / histogram->interval_size_usec();
  if (interval_index > histogram->num_intervals() - 1) {
    interval_index = histogram->num_intervals() - 1;
  }
  histogram->set_count(interval_index, histogram->count(interval_index) + 1);
}

int64 GraphProfiler::AddInputStreamTimeSamples(
    const CalculatorContext& calculator_context, int64 start_time_usec,
    CalculatorProfile* calculator_profile) {
  int64 input_timestamp_usec = calculator_context.InputTimestamp().Value();
  int64 min_source_process_start_usec = start_time_usec;
  int64 input_stream_counter = -1;
  for (CollectionItemId id = calculator_context.Inputs().BeginId();
       id < calculator_context.Inputs().EndId(); ++id) {
    ++input_stream_counter;
    if (calculator_context.Inputs().Get(id).Value().IsEmpty() ||
        calculator_profile->input_stream_profiles(input_stream_counter)
            .back_edge()) {
      continue;
    }

    PacketId packet_id = {calculator_context.Inputs().Get(id).Name(),
                          input_timestamp_usec};
    PacketInfo* packet_info = GetPacketInfo(&packets_info_, packet_id);
    if (packet_info == nullptr) {
      // This is a condition rather than a failure CHECK because
      // under certain conditions the consumer calculator's Process()
      // can start before the producer calculator's Process() is finished.
      LOG_FIRST_N(WARNING, 10) << "Expected packet info is missing for: "
                               << PacketIdToString(packet_id);
      continue;
    }
    AddTimeSample(
        packet_info->production_time_usec, start_time_usec,
        calculator_profile->mutable_input_stream_profiles(input_stream_counter)
            ->mutable_latency());

    min_source_process_start_usec = std::min(
        min_source_process_start_usec, packet_info->source_process_start_usec);
  }

  return min_source_process_start_usec;
}

void GraphProfiler::AddProcessSample(
    const CalculatorContext& calculator_context, int64 start_time_usec,
    int64 end_time_usec) {
  absl::ReaderMutexLock lock(&profiler_mutex_);
  if (!is_profiling_) {
    return;
  }

  const std::string& node_name = calculator_context.NodeName();
  auto profile_iter = calculator_profiles_.find(node_name);
  CHECK(profile_iter != calculator_profiles_.end()) << absl::Substitute(
      "Calculator \"$0\" has not been added during initialization.",
      calculator_context.NodeName());
  CalculatorProfile* calculator_profile = &profile_iter->second;

  // Update Process() runtime.
  AddTimeSample(start_time_usec, end_time_usec,
                calculator_profile->mutable_process_runtime());

  if (profiler_config_.enable_stream_latency()) {
    int64 min_source_process_start_usec = AddStreamLatencies(
        calculator_context, start_time_usec, end_time_usec, calculator_profile);
    // Update input and output trace latencies.
    AddTimeSample(min_source_process_start_usec, start_time_usec,
                  calculator_profile->mutable_process_input_latency());
    AddTimeSample(min_source_process_start_usec, end_time_usec,
                  calculator_profile->mutable_process_output_latency());
  }
}

std::unique_ptr<GlProfilingHelper> GraphProfiler::CreateGlProfilingHelper() {
  if (!IsTracerEnabled(profiler_config_)) {
    return nullptr;
  }
  return absl::make_unique<mediapipe::GlProfilingHelper>(shared_from_this());
}

// A simple ZeroCopyOutputStream that writes to a std::ostream.
class OstreamStream : public proto_ns::io::ZeroCopyOutputStream {
 public:
  explicit OstreamStream(std::ostream* output)
      : output_(output), buffer_used_(0), position_(0) {}
  ~OstreamStream() override { WriteBuffer(); }
  bool Next(void** data, int* size) override {
    *data = buffer_;
    *size = sizeof(buffer_);
    return WriteBuffer();
  }
  void BackUp(int count) override { buffer_used_ -= count; }
  int64_t ByteCount() const override { return position_; }

 private:
  // Writes the buffer to the ostream.
  bool WriteBuffer() {
    output_->write(buffer_, buffer_used_);
    position_ += buffer_used_;
    buffer_used_ = sizeof(buffer_);
    return output_->good();
  }
  std::ostream* output_;
  char buffer_[1024];
  int buffer_used_;
  int position_;
  OstreamStream(const OstreamStream&) = delete;
  OstreamStream& operator=(const OstreamStream&) = delete;
};

// Sets the canonical node name in each CalculatorGraphConfig::Node
// and also in the GraphTrace if present.
void AssignNodeNames(GraphProfile* profile) {
  CalculatorGraphConfig* graph_config = profile->mutable_config();
  GraphTrace* graph_trace = profile->graph_trace_size() > 0
                                ? profile->mutable_graph_trace(0)
                                : nullptr;
  if (graph_trace) {
    graph_trace->clear_calculator_name();
  }
  std::vector<std::string> canonical_names;
  canonical_names.reserve(graph_config->node().size());
  for (int i = 0; i < graph_config->node().size(); ++i) {
    canonical_names.push_back(CanonicalNodeName(*graph_config, i));
  }
  for (int i = 0; i < graph_config->node().size(); ++i) {
    graph_config->mutable_node(i)->set_name(canonical_names[i]);
  }
  if (graph_trace) {
    graph_trace->mutable_calculator_name()->Assign(canonical_names.begin(),
                                                   canonical_names.end());
  }
}

// Clears fields containing their default values.
void CleanTimeHistogram(TimeHistogram* histogram) {
  if (histogram->num_intervals() == 1) {
    histogram->clear_num_intervals();
  }
  if (histogram->interval_size_usec() == 1000000) {
    histogram->clear_interval_size_usec();
  }
}

// Clears fields containing their default values.
void CleanCalculatorProfiles(GraphProfile* profile) {
  for (CalculatorProfile& p : *profile->mutable_calculator_profiles()) {
    CleanTimeHistogram(p.mutable_process_runtime());
    CleanTimeHistogram(p.mutable_process_input_latency());
    CleanTimeHistogram(p.mutable_process_output_latency());
    for (StreamProfile& s : *p.mutable_input_stream_profiles()) {
      CleanTimeHistogram(s.mutable_latency());
    }
  }
}

absl::StatusOr<std::string> GraphProfiler::GetTraceLogPath() {
  if (!IsTraceLogEnabled(profiler_config_)) {
    return absl::InternalError(
        "Trace log writing is disabled, unable to get trace_log_path.");
  }
  if (profiler_config_.trace_log_path().empty()) {
    ASSIGN_OR_RETURN(std::string directory_path, GetDefaultTraceLogDirectory());
    std::string trace_log_path =
        absl::StrCat(directory_path, "/", kDefaultLogFilePrefix);
    return trace_log_path;
  } else {
    return profiler_config_.trace_log_path();
  }
}

absl::Status GraphProfiler::CaptureProfile(
    GraphProfile* result, PopulateGraphConfig populate_config) {
  // Record the GraphTrace events since the previous WriteProfile.
  // The end_time is chosen to be trace_log_margin_usec in the past,
  // providing time for events to be appended to the TraceBuffer.
  absl::Time end_time =
      clock_->TimeNow() -
      absl::Microseconds(profiler_config_.trace_log_margin_usec());
  if (tracer()) {
    GraphTrace* trace = result->add_graph_trace();
    if (!profiler_config_.trace_log_instant_events()) {
      tracer()->GetTrace(previous_log_end_time_, end_time, trace);
    } else {
      tracer()->GetLog(previous_log_end_time_, end_time, trace);
    }
  }
  previous_log_end_time_ = end_time;

  // Record the latest CalculatorProfiles.
  Status status;
  std::vector<CalculatorProfile> profiles;
  status.Update(GetCalculatorProfiles(&profiles));
  for (CalculatorProfile& p : profiles) {
    if (profile_builder_->ProfileIncluded(p)) {
      *result->mutable_calculator_profiles()->Add() = std::move(p);
    }
  }
  this->Reset();
  CleanCalculatorProfiles(result);
  if (populate_config == PopulateGraphConfig::kFull) {
    *result->mutable_config() = validated_graph_->Config();
    AssignNodeNames(result);
  }
  return status;
}

absl::Status GraphProfiler::WriteProfile() {
  if (profiler_config_.trace_log_disabled()) {
    // Logging is disabled, so we can exit writing without error.
    return absl::OkStatus();
  }
  ASSIGN_OR_RETURN(std::string trace_log_path, GetTraceLogPath());
  int log_interval_count = GetLogIntervalCount(profiler_config_);
  int log_file_count = GetLogFileCount(profiler_config_);
  GraphProfile profile;
  MP_RETURN_IF_ERROR(CaptureProfile(&profile, PopulateGraphConfig::kNo));

  // If there are no trace events, skip log writing.
  const GraphTrace& trace = *profile.graph_trace().rbegin();
  if (is_tracing_ && trace.calculator_trace().empty()) {
    return absl::OkStatus();
  }

  // Record the CalculatorGraphConfig, once per log file.
  ++previous_log_index_;
  bool is_new_file = (previous_log_index_ % log_interval_count == 0);
  if (is_new_file) {
    *profile.mutable_config() = validated_graph_->Config();
    AssignNodeNames(&profile);
  }

  // Write the GraphProfile to the trace_log_path.
  int log_index = previous_log_index_ / log_interval_count % log_file_count;
  std::string log_path = absl::StrCat(trace_log_path, log_index, ".binarypb");
  std::ofstream ofs;
  if (is_new_file) {
    ofs.open(log_path, std::ofstream::out | std::ofstream::trunc);
  } else {
    ofs.open(log_path, std::ofstream::out | std::ofstream::app);
  }
  OstreamStream out(&ofs);
  RET_CHECK(profile.SerializeToZeroCopyStream(&out))
      << "Could not write binary GraphProfile to: " << log_path;
  return absl::OkStatus();
}

}  // namespace mediapipe
