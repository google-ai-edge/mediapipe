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

#include "mediapipe/util/tracking/box_tracker.h"

#include <sys/stat.h>

#include <fstream>
#include <limits>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/tracking.pb.h"

namespace mediapipe {

// Time within which close checkpoints are removed.
static constexpr int kSnapMs = 1000;
static constexpr int kInitCheckpoint = -1;

void MotionBoxStateQuadToVertices(const MotionBoxState::Quad& quad,
                                  std::vector<Vector2_f>* vertices) {
  CHECK_EQ(TimedBox::kNumQuadVertices * 2, quad.vertices_size());
  CHECK(vertices != nullptr);
  vertices->clear();
  for (int i = 0; i < TimedBox::kNumQuadVertices; ++i) {
    vertices->push_back(
        Vector2_f(quad.vertices(i * 2), quad.vertices(i * 2 + 1)));
  }
}

void VerticesToMotionBoxStateQuad(const std::vector<Vector2_f>& vertices,
                                  MotionBoxState::Quad* quad) {
  CHECK_EQ(TimedBox::kNumQuadVertices, vertices.size());
  CHECK(quad != nullptr);
  for (const Vector2_f& vertex : vertices) {
    quad->add_vertices(vertex.x());
    quad->add_vertices(vertex.y());
  }
}

void MotionBoxStateFromTimedBox(const TimedBox& box, MotionBoxState* state) {
  CHECK(state);
  state->set_pos_x(box.left);
  state->set_pos_y(box.top);
  state->set_width(box.right - box.left);
  state->set_height(box.bottom - box.top);
  state->set_rotation(box.rotation);
  state->set_request_grouping(box.request_grouping);
  if (box.quad_vertices.size() == TimedBox::kNumQuadVertices) {
    VerticesToMotionBoxStateQuad(box.quad_vertices, state->mutable_quad());

    if (box.aspect_ratio > 0.0f) {
      state->set_aspect_ratio(box.aspect_ratio);
    }

    // set pos_x and pos_y to be the top-left vertex x and y coordinates
    // set width and height to be max - min of x and y.
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& vertex : box.quad_vertices) {
      min_x = std::min(min_x, vertex.x());
      max_x = std::max(max_x, vertex.x());
      min_y = std::min(min_y, vertex.y());
      max_y = std::max(max_y, vertex.y());
    }
    state->set_pos_x(min_x);
    state->set_pos_y(min_y);
    state->set_width(max_x - min_x);
    state->set_height(max_y - min_y);
  }
}

void TimedBoxFromMotionBoxState(const MotionBoxState& state, TimedBox* box) {
  CHECK(box);
  const float scale_dx = state.width() * (state.scale() - 1.0f) * 0.5f;
  const float scale_dy = state.height() * (state.scale() - 1.0f) * 0.5f;
  box->left = state.pos_x() - scale_dx;
  box->top = state.pos_y() - scale_dy;
  box->right = state.pos_x() + state.width() + scale_dx;
  box->bottom = state.pos_y() + state.height() + scale_dy;
  box->rotation = state.rotation();
  box->confidence = state.tracking_confidence();
  box->request_grouping = state.request_grouping();
  if (state.has_quad()) {
    MotionBoxStateQuadToVertices(state.quad(), &(box->quad_vertices));

    if (state.has_aspect_ratio()) {
      box->aspect_ratio = state.aspect_ratio();
    }
  }
}

namespace {

TimedBox BlendTimedBoxes(const TimedBox& lhs, const TimedBox& rhs,
                         int64 time_msec) {
  CHECK_LT(lhs.time_msec, rhs.time_msec);
  const double alpha =
      (time_msec - lhs.time_msec) * 1.0 / (rhs.time_msec - lhs.time_msec);
  return TimedBox::Blend(lhs, rhs, alpha);
}

}  // namespace.

TimedBox TimedBox::Blend(const TimedBox& lhs, const TimedBox& rhs, double alpha,
                         double beta) {
  // Due to large timestamps alpha beta should be in double.
  TimedBox result;
  result.top = alpha * lhs.top + beta * rhs.top;
  result.left = alpha * lhs.left + beta * rhs.left;
  result.bottom = alpha * lhs.bottom + beta * rhs.bottom;
  result.right = alpha * lhs.right + beta * rhs.right;
  result.rotation = alpha * lhs.rotation + beta * rhs.rotation;
  result.time_msec = std::round(alpha * lhs.time_msec + beta * rhs.time_msec);
  result.confidence = alpha * lhs.confidence + beta * rhs.confidence;
  if (lhs.quad_vertices.size() == kNumQuadVertices &&
      rhs.quad_vertices.size() == kNumQuadVertices) {
    result.quad_vertices.clear();
    for (int i = 0; i < lhs.quad_vertices.size(); ++i) {
      result.quad_vertices.push_back(alpha * lhs.quad_vertices[i] +
                                     beta * rhs.quad_vertices[i]);
    }

    // Since alpha and beta are not necessarily sum to 1, aspect ratio can not
    // be derived with alpha and beta. Here we are simply averaging the
    // aspect_ratio as the blended box aspect ratio.
    if (lhs.aspect_ratio > 0 && rhs.aspect_ratio > 0) {
      result.aspect_ratio = 0.5f * lhs.aspect_ratio + 0.5f * rhs.aspect_ratio;
    }
  }
  return result;
}

TimedBox TimedBox::Blend(const TimedBox& lhs, const TimedBox& rhs,
                         double alpha) {
  return Blend(lhs, rhs, 1.0 - alpha, alpha);
}

std::array<Vector2_f, 4> TimedBox::Corners(float width, float height) const {
  if (quad_vertices.size() == kNumQuadVertices) {
    std::array<Vector2_f, 4> corners{{
        Vector2_f(quad_vertices[0].x() * width, quad_vertices[0].y() * height),
        Vector2_f(quad_vertices[1].x() * width, quad_vertices[1].y() * height),
        Vector2_f(quad_vertices[2].x() * width, quad_vertices[2].y() * height),
        Vector2_f(quad_vertices[3].x() * width, quad_vertices[3].y() * height),
    }};
    return corners;
  } else {
    // Rotate 4 corner w.r.t. center.
    const Vector2_f center(0.5f * (left + right) * width,
                           0.5f * (top + bottom) * height);
    const std::array<Vector2_f, 4> corners{{
        Vector2_f(left * width, top * height),
        Vector2_f(left * width, bottom * height),
        Vector2_f(right * width, bottom * height),
        Vector2_f(right * width, top * height),
    }};

    const float cos_a = std::cos(rotation);
    const float sin_a = std::sin(rotation);
    std::array<Vector2_f, 4> transformed_corners;
    for (int k = 0; k < 4; ++k) {
      // Scale and rotate w.r.t. center.
      const Vector2_f rad = corners[k] - center;
      const Vector2_f rot_rad(cos_a * rad.x() - sin_a * rad.y(),
                              sin_a * rad.x() + cos_a * rad.y());
      transformed_corners[k] = center + rot_rad;
    }
    return transformed_corners;
  }
}

TimedBox TimedBox::FromProto(const TimedBoxProto& proto) {
  TimedBox box;
  box.top = proto.top();
  box.left = proto.left();
  box.bottom = proto.bottom();
  box.right = proto.right();
  box.rotation = proto.rotation();
  box.time_msec = proto.time_msec();
  box.request_grouping = proto.request_grouping();
  if (proto.has_quad() &&
      proto.quad().vertices_size() == kNumQuadVertices * 2) {
    MotionBoxStateQuadToVertices(proto.quad(), &(box.quad_vertices));

    if (proto.has_aspect_ratio()) {
      box.aspect_ratio = proto.aspect_ratio();
    }
  }
  return box;
}

TimedBoxProto TimedBox::ToProto() const {
  TimedBoxProto proto;
  proto.set_top(top);
  proto.set_left(left);
  proto.set_bottom(bottom);
  proto.set_right(right);
  proto.set_rotation(rotation);
  proto.set_time_msec(time_msec);
  proto.set_confidence(confidence);
  proto.set_request_grouping(request_grouping);
  if (quad_vertices.size() == kNumQuadVertices) {
    VerticesToMotionBoxStateQuad(quad_vertices, proto.mutable_quad());

    if (aspect_ratio > 0.0f) {
      proto.set_aspect_ratio(aspect_ratio);
    }
  }
  return proto;
}

BoxTracker::BoxTracker(const std::string& cache_dir,
                       const BoxTrackerOptions& options)
    : options_(options), cache_dir_(cache_dir) {
  tracking_workers_.reset(new ThreadPool(options_.num_tracking_workers()));
  tracking_workers_->StartWorkers();
}

BoxTracker::BoxTracker(
    const std::vector<const TrackingDataChunk*>& tracking_data, bool copy_data,
    const BoxTrackerOptions& options)
    : BoxTracker("", options) {
  AddTrackingDataChunks(tracking_data, copy_data);
}

void BoxTracker::AddTrackingDataChunk(const TrackingDataChunk* chunk,
                                      bool copy_data) {
  CHECK_GT(chunk->item_size(), 0) << "Empty chunk.";
  int64 chunk_time_msec = chunk->item(0).timestamp_usec() / 1000;
  int chunk_idx = ChunkIdxFromTime(chunk_time_msec);
  CHECK_GE(chunk_idx, tracking_data_.size()) << "Chunk is out of order.";
  if (chunk_idx > tracking_data_.size()) {
    LOG(INFO) << "Resize tracking_data_ to " << chunk_idx;
    tracking_data_.resize(chunk_idx);
  }
  if (copy_data) {
    tracking_data_buffer_.emplace_back(new TrackingDataChunk(*chunk));
    tracking_data_.push_back(tracking_data_buffer_.back().get());
  } else {
    tracking_data_.emplace_back(chunk);
  }
}

void BoxTracker::AddTrackingDataChunks(
    const std::vector<const TrackingDataChunk*>& tracking_data,
    bool copy_data) {
  for (const auto item : tracking_data) {
    AddTrackingDataChunk(item, copy_data);
  }
}

void BoxTracker::NewBoxTrack(const TimedBox& initial_pos, int id,
                             int64 min_msec, int64 max_msec) {
  VLOG(1) << "New box track: " << id << " : " << initial_pos.ToString()
          << " from " << min_msec << " to " << max_msec;

  // Mark initialization with checkpoint -1.
  absl::MutexLock lock(&status_mutex_);

  if (canceling_) {
    LOG(WARNING) << "Box Tracker is in cancel state. Refusing request.";
    return;
  }
  ++track_status_[id][kInitCheckpoint].tracks_ongoing;

  auto operation = [this, initial_pos, id, min_msec, max_msec]() {
    this->NewBoxTrackAsync(initial_pos, id, min_msec, max_msec);
  };

  tracking_workers_->Schedule(operation);
}

std::pair<int64, int64> BoxTracker::TrackInterval(int id) {
  absl::MutexLock lock(&path_mutex_);
  const Path& path = paths_[id];
  if (path.empty()) {
    return std::make_pair(-1, -1);
  }

  auto first_interval = path.begin()->second;
  auto last_interval = path.rbegin()->second;

  return std::make_pair(first_interval.front().time_msec,
                        last_interval.back().time_msec);
}

void BoxTracker::NewBoxTrackAsync(const TimedBox& initial_pos, int id,
                                  int64 min_msec, int64 max_msec) {
  VLOG(1) << "Async track for id: " << id << " from " << min_msec << " to "
          << max_msec;

  // Determine start position and track forward and backward.
  int chunk_idx = ChunkIdxFromTime(initial_pos.time_msec);

  VLOG(1) << "Starting at chunk " << chunk_idx;

  AugmentedChunkPtr tracking_chunk(ReadChunk(id, kInitCheckpoint, chunk_idx));

  if (!tracking_chunk.first) {
    absl::MutexLock lock(&status_mutex_);
    --track_status_[id][kInitCheckpoint].tracks_ongoing;
    LOG(ERROR) << "Could not read tracking chunk from file: " << chunk_idx
               << " for start position: " << initial_pos.ToString();
    return;
  }

  // Grab ownership here, to avoid any memory leaks due to early return.
  std::unique_ptr<const TrackingDataChunk> chunk_owned;
  if (tracking_chunk.second) {
    chunk_owned.reset(tracking_chunk.first);
  }

  const int start_frame =
      ClosestFrameIndex(initial_pos.time_msec, *tracking_chunk.first);

  VLOG(1) << "Local start frame: " << start_frame;

  // Update starting position to coincide with a frame.
  TimedBox start_pos = initial_pos;
  start_pos.time_msec =
      tracking_chunk.first->item(start_frame).timestamp_usec() / 1000;

  VLOG(1) << "Request at " << initial_pos.time_msec << " revised to "
          << start_pos.time_msec;

  const int checkpoint = start_pos.time_msec;

  // TODO:
  // Compute min and max for tracking based on existing check points.
  if (!WaitToScheduleId(id)) {
    // Could not schedule, id already being canceled.
    return;
  }

  // If another checkpoint is close by, cancel that one.
  VLOG(1) << "Removing close checkpoints";
  absl::MutexLock lock(&status_mutex_);
  RemoveCloseCheckpoints(id, checkpoint);

  VLOG(1) << "Cancel existing tracks";
  CancelTracking(id, checkpoint);

  // Remove checkpoint results (to be replaced with current one).
  ClearCheckpoint(id, checkpoint);

  MotionBoxState start_state;
  MotionBoxStateFromTimedBox(start_pos, &start_state);

  VLOG(1) << "Adding initial result";
  AddBoxResult(start_pos, id, checkpoint, start_state);

  // Perform forward and backward tracking and add to current PathSegment.
  // Track forward.
  track_status_[id][checkpoint].tracks_ongoing += 2;

  VLOG(1) << "Starting tracking workers ... ";

  AugmentedChunkPtr forward_chunk = tracking_chunk;
  AugmentedChunkPtr backward_chunk = tracking_chunk;

  if (tracking_chunk.second) {  // We have ownership, need a copy here.
    forward_chunk = std::make_pair(new TrackingDataChunk(*chunk_owned), true);
    backward_chunk = std::make_pair(chunk_owned.release(), true);
  }

  auto forward_operation = [this, forward_chunk, start_state, start_frame,
                            chunk_idx, id, checkpoint, min_msec, max_msec]() {
    this->TrackingImpl(TrackingImplArgs(forward_chunk, start_state, start_frame,
                                        chunk_idx, id, checkpoint, true, true,
                                        min_msec, max_msec));
  };

  tracking_workers_->Schedule(forward_operation);

  // Track backward.
  auto backward_operation = [this, backward_chunk, start_state, start_frame,
                             chunk_idx, id, checkpoint, min_msec, max_msec]() {
    this->TrackingImpl(TrackingImplArgs(backward_chunk, start_state,
                                        start_frame, chunk_idx, id, checkpoint,
                                        false, true, min_msec, max_msec));
  };

  tracking_workers_->Schedule(backward_operation);

  DoneSchedulingId(id);

  // Tell a waiting request that we are done scheduling.
  status_condvar_.SignalAll();
  VLOG(1) << "Scheduling done for " << id;
}

void BoxTracker::RemoveCloseCheckpoints(int id, int checkpoint) {
  if (track_status_[id].empty()) {
    return;
  }

  auto pos = track_status_[id].lower_bound(checkpoint);
  // Test current and previous location (if possible).
  int num_turns = 1;
  if (pos != track_status_[id].begin()) {
    --pos;
    ++num_turns;
  }

  for (int k = 0; k < num_turns; ++k, ++pos) {
    if (pos != track_status_[id].end()) {
      const int check_pos = pos->first;
      // Ignore marker init checkpoint from track_status_.
      if (check_pos > kInitCheckpoint &&
          std::abs(check_pos - checkpoint) < kSnapMs) {
        CancelTracking(id, check_pos);
        ClearCheckpoint(id, check_pos);
      }
    } else {
      break;
    }
  }
}

bool BoxTracker::WaitToScheduleId(int id) {
  VLOG(1) << "Wait to schedule id: " << id;
  absl::MutexLock lock(&status_mutex_);
  while (new_box_track_[id]) {
    // Box tracking is currently ongoing for this id.
    if (track_status_[id][kInitCheckpoint].canceled) {
      // Canceled, remove myself from ongoing tracks.
      --track_status_[id][kInitCheckpoint].tracks_ongoing;
      status_condvar_.SignalAll();
      return false;
    }

    // Only one request can be processing in the section till end of the
    // function NewBoxTrackAsync at a time.
    status_condvar_.Wait(&status_mutex_);
  }

  // We got canceled already, don't proceed.
  if (track_status_[id][kInitCheckpoint].canceled) {
    --track_status_[id][kInitCheckpoint].tracks_ongoing;
    status_condvar_.SignalAll();
    return false;
  }

  // Signal we are about to schedule new tracking.
  new_box_track_[id] = true;
  VLOG(1) << "Ready to schedule id:  " << id;
  return true;
}

void BoxTracker::DoneSchedulingId(int id) {
  new_box_track_[id] = false;
  --track_status_[id][kInitCheckpoint].tracks_ongoing;
}

void BoxTracker::CancelTracking(int id, int checkpoint) {
  // Wait for ongoing requests to terminate.
  while (track_status_[id][checkpoint].tracks_ongoing != 0) {
    // Cancel all ongoing requests.
    track_status_[id][checkpoint].canceled = true;
    status_condvar_.Wait(&status_mutex_);
  }

  track_status_[id][checkpoint].canceled = false;
}

bool BoxTracker::GetTimedPosition(int id, int64 time_msec, TimedBox* result,
                                  std::vector<MotionBoxState>* states) {
  CHECK(result);

  MotionBoxState* lhs_box_state = nullptr;
  MotionBoxState* rhs_box_state = nullptr;
  if (states) {
    CHECK(options_.record_path_states())
        << "Requesting corresponding tracking states requires option "
        << "record_path_states to be set";
    states->resize(1);
    lhs_box_state = rhs_box_state = &states->at(0);
  }

  VLOG(1) << "Obtaining result at " << time_msec;

  absl::MutexLock lock(&path_mutex_);
  const Path& path = paths_[id];
  if (path.empty()) {
    LOG(ERROR) << "Empty path!";
    return false;
  }

  // Find corresponding checkpoint.
  auto check_pos = path.lower_bound(time_msec);
  if (check_pos == path.begin()) {
    VLOG(1) << "To left";
    // We are to the left of the earliest checkpoint.
    return TimedBoxAtTime(check_pos->second, time_msec, result, lhs_box_state);
  }
  if (check_pos == path.end()) {
    VLOG(1) << "To right";
    --check_pos;
    // We are to the right of the lastest checkpoint.
    return TimedBoxAtTime(check_pos->second, time_msec, result, rhs_box_state);
  }

  VLOG(1) << "Blending ...";

  // We are inbetween checkpoints, get result for each, then blend.
  const PathSegment& rhs = check_pos->second;
  const int check_rhs = check_pos->first;
  --check_pos;
  const PathSegment& lhs = check_pos->second;
  const int check_lhs = check_pos->first;

  TimedBox lhs_box;
  TimedBox rhs_box;
  if (states) {
    states->resize(2);
    lhs_box_state = &states->at(0);
    rhs_box_state = &states->at(1);
  }

  if (!TimedBoxAtTime(lhs, time_msec, &lhs_box, lhs_box_state)) {
    return false;
  }

  if (!TimedBoxAtTime(rhs, time_msec, &rhs_box, rhs_box_state)) {
    return false;
  }

  VLOG(1) << "Blending: " << lhs_box.ToString() << " and "
          << rhs_box.ToString();
  const double alpha = (time_msec - check_lhs) * 1.0 / (check_rhs - check_lhs);
  *result = TimedBox::Blend(lhs_box, rhs_box, alpha);

  return true;
}

bool BoxTracker::IsTrackingOngoingForId(int id) {
  absl::MutexLock lock(&status_mutex_);
  for (const auto& item : track_status_[id]) {
    if (item.second.tracks_ongoing > 0) {
      return true;
    }
  }
  return false;
}

bool BoxTracker::IsTrackingOngoing() {
  absl::MutexLock lock(&status_mutex_);
  return IsTrackingOngoingMutexHeld();
}

bool BoxTracker::IsTrackingOngoingMutexHeld() {
  for (const auto& id : track_status_) {
    for (const auto& item : id.second) {
      if (item.second.tracks_ongoing > 0) {
        return true;
      }
    }
  }
  return false;
}

BoxTracker::AugmentedChunkPtr BoxTracker::ReadChunk(int id, int checkpoint,
                                                    int chunk_idx) {
  VLOG(1) << __FUNCTION__ << " id=" << id << " chunk_idx=" << chunk_idx;
  if (cache_dir_.empty() && !tracking_data_.empty()) {
    if (chunk_idx < tracking_data_.size()) {
      return std::make_pair(tracking_data_[chunk_idx], false);
    } else {
      LOG(ERROR) << "chunk_idx >= tracking_data_.size()";
      return std::make_pair(nullptr, false);
    }
  } else {
    std::unique_ptr<TrackingDataChunk> chunk_data(
        ReadChunkFromCache(id, checkpoint, chunk_idx));
    return std::make_pair(chunk_data.release(), true);
  }
}

std::unique_ptr<TrackingDataChunk> BoxTracker::ReadChunkFromCache(
    int id, int checkpoint, int chunk_idx) {
  VLOG(1) << __FUNCTION__ << " id=" << id << " chunk_idx=" << chunk_idx;

  auto format_runtime =
      absl::ParsedFormat<'d'>::New(options_.cache_file_format());

  std::string chunk_file;
  if (format_runtime) {
    chunk_file = cache_dir_ + "/" + absl::StrFormat(*format_runtime, chunk_idx);
  } else {
    LOG(ERROR) << "chache_file_format wrong. fall back to chunk_%04d.";
    chunk_file = cache_dir_ + "/" + absl::StrFormat("chunk_%04d", chunk_idx);
  }

  VLOG(1) << "Reading chunk from cache: " << chunk_file;
  std::unique_ptr<TrackingDataChunk> chunk_data(new TrackingDataChunk());

  struct stat tmp;
  if (stat(chunk_file.c_str(), &tmp)) {
    if (!WaitForChunkFile(id, checkpoint, chunk_file)) {
      return nullptr;
    }
  }

  VLOG(1) << "File exists, reading ...";

  std::ifstream in(chunk_file, std::ios::in | std::ios::binary);
  if (!in) {
    LOG(ERROR) << "Could not read chunk file: " << chunk_file;
    return nullptr;
  }

  std::string data;
  in.seekg(0, std::ios::end);
  data.resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&data[0], data.size());
  in.close();

  chunk_data->ParseFromString(data);

  VLOG(1) << "Read success";
  return chunk_data;
}

bool BoxTracker::WaitForChunkFile(int id, int checkpoint,
                                  const std::string& chunk_file) {
  VLOG(1) << "Chunk no exists, waiting for file: " << chunk_file;

  const int timeout_msec = options_.read_chunk_timeout_msec();

  // Exponential backoff sleep till file exists.
  int wait_time_msec = 20;

  VLOG(1) << "In wait for chunk ...: " << chunk_file;

  // Maximum wait time.
  const int kMaxWaitPeriod = 5000;
  int total_wait_msec = 0;
  bool file_exists = false;

  while (!file_exists && total_wait_msec < timeout_msec) {
    // Check if we got canceled.
    {
      absl::MutexLock lock(&status_mutex_);
      if (track_status_[id][checkpoint].canceled) {
        return false;
      }
    }

    absl::SleepFor(absl::Milliseconds(wait_time_msec));
    total_wait_msec += wait_time_msec;

    struct stat tmp;
    file_exists = stat(chunk_file.c_str(), &tmp) == 0;

    if (file_exists) {
      VLOG(1) << "Sucessfully waited on " << chunk_file << " for "
              << total_wait_msec;
      break;
    }
    if (wait_time_msec < kMaxWaitPeriod) {
      wait_time_msec *= 1.5;
    }
  }

  return file_exists;
}

int BoxTracker::ClosestFrameIndex(int64 msec,
                                  const TrackingDataChunk& chunk) const {
  CHECK_GT(chunk.item_size(), 0);
  typedef TrackingDataChunk::Item Item;
  Item item_to_find;
  item_to_find.set_timestamp_usec(msec * 1000);
  int pos =
      std::lower_bound(chunk.item().begin(), chunk.item().end(), item_to_find,
                       [](const Item& lhs, const Item& rhs) -> bool {
                         return lhs.timestamp_usec() < rhs.timestamp_usec();
                       }) -
      chunk.item().begin();

  // Skip end.
  if (pos == chunk.item_size()) {
    return pos - 1;
  } else if (pos == 0) {
    // Nothing smaller exists.
    return 0;
  }

  // Determine closest timestamp.
  const int64 lhs_diff = msec - chunk.item(pos - 1).timestamp_usec() / 1000;
  const int64 rhs_diff = chunk.item(pos).timestamp_usec() / 1000 - msec;

  if (std::min(lhs_diff, rhs_diff) >= 67) {
    LOG(ERROR) << "No frame found within 67ms, probably using wrong chunk.";
  }

  if (lhs_diff < rhs_diff) {
    return pos - 1;
  } else {
    return pos;
  }
}

void BoxTracker::AddBoxResult(const TimedBox& box, int id, int checkpoint,
                              const MotionBoxState& state) {
  absl::MutexLock lock(&path_mutex_);
  PathSegment& segment = paths_[id][checkpoint];
  auto insert_pos = std::lower_bound(segment.begin(), segment.end(), box);
  const bool store_state = options_.record_path_states();

  // Don't overwrite an existing box.
  if (insert_pos == segment.end() || insert_pos->time_msec != box.time_msec) {
    segment.insert(insert_pos,
                   InternalTimedBox(
                       box, store_state ? new MotionBoxState(state) : nullptr));
  }
}

void BoxTracker::ClearCheckpoint(int id, int checkpoint) {
  absl::MutexLock lock(&path_mutex_);
  PathSegment& segment = paths_[id][checkpoint];
  segment.clear();
}

void BoxTracker::TrackingImpl(const TrackingImplArgs& a) {
  TrackStepOptions track_step_options = options_.track_step_options();
  ChangeTrackingDegreesBasedOnStartPos(a.start_state, &track_step_options);
  MotionBox motion_box(track_step_options);
  const int chunk_data_size = a.chunk_data->item_size();

  CHECK_GE(a.start_frame, 0);
  CHECK_LT(a.start_frame, chunk_data_size);

  VLOG(1) << " a.start_frame = " << a.start_frame << " @"
          << a.chunk_data->item(a.start_frame).timestamp_usec() << " with "
          << chunk_data_size << " items";
  motion_box.ResetAtFrame(a.start_frame, a.start_state);

  auto cleanup_func = [&a, this]() -> void {
    if (a.first_call) {
      // Signal we are done processing in this direction.
      absl::MutexLock lock(&status_mutex_);
      --track_status_[a.id][a.checkpoint].tracks_ongoing;
      status_condvar_.SignalAll();
    }
  };

  if (a.forward) {
    // TrackingData at frame f, contains tracking information from
    // frame f to f - 1. Get information at frame f + 1 and invert:
    // Tracking from f to f + 1.
    for (int f = a.start_frame; f + 1 < chunk_data_size; ++f) {
      // Note: we use / 1000 instead of * 1000 to avoid overflow.
      if (a.chunk_data->item(f + 1).timestamp_usec() / 1000 > a.max_msec) {
        VLOG(2) << "Reached maximum tracking timestamp @" << a.max_msec;
        break;
      }
      VLOG(1) << "Track forward from " << f;
      MotionVectorFrame mvf;
      MotionVectorFrameFromTrackingData(
          a.chunk_data->item(f + 1).tracking_data(), &mvf);
      const int track_duration_ms =
          TrackingDataDurationMs(a.chunk_data->item(f + 1));
      if (track_duration_ms > 0) {
        mvf.duration_ms = track_duration_ms;
      }

      // If this is the first frame in a chunk, there might be an unobserved
      // chunk boundary at the first frame.
      if (f == 0 && a.chunk_data->item(0).tracking_data().frame_flags() &
                        TrackingData::FLAG_CHUNK_BOUNDARY) {
        mvf.is_chunk_boundary = true;
      }

      MotionVectorFrame mvf_inverted;
      InvertMotionVectorFrame(mvf, &mvf_inverted);

      const bool forward_tracking = true;
      if (!motion_box.TrackStep(f, mvf_inverted, forward_tracking)) {
        VLOG(1) << "Failed forward at frame: " << f;
        break;
      } else {
        // Test if current request is canceled.
        {
          absl::MutexLock lock(&status_mutex_);
          if (track_status_[a.id][a.checkpoint].canceled) {
            --track_status_[a.id][a.checkpoint].tracks_ongoing;
            status_condvar_.SignalAll();
            return;
          }
        }

        TimedBox result;
        const MotionBoxState& result_state = motion_box.StateAtFrame(f + 1);
        TimedBoxFromMotionBoxState(result_state, &result);
        result.time_msec = a.chunk_data->item(f + 1).timestamp_usec() / 1000;
        AddBoxResult(result, a.id, a.checkpoint, result_state);
      }

      if (f + 2 == chunk_data_size && !a.chunk_data->last_chunk()) {
        // Last frame, successful track, continue;
        AugmentedChunkPtr next_chunk(
            ReadChunk(a.id, a.checkpoint, a.chunk_idx + 1));

        if (next_chunk.first != nullptr) {
          TrackingImplArgs next_args(next_chunk, motion_box.StateAtFrame(f + 1),
                                     0, a.chunk_idx + 1, a.id, a.checkpoint,
                                     a.forward, false, a.min_msec, a.max_msec);

          TrackingImpl(next_args);
        } else {
          cleanup_func();
          LOG(ERROR) << "Can't read expected chunk file!";
        }
      }
    }
  } else {
    // Backward tracking.
    // Don't attempt to track from the very first frame backwards.
    const int first_frame = a.chunk_data->first_chunk() ? 1 : 0;

    for (int f = a.start_frame; f >= first_frame; --f) {
      if (a.chunk_data->item(f).timestamp_usec() / 1000 < a.min_msec) {
        VLOG(2) << "Reached minimum tracking timestamp @" << a.min_msec;
        break;
      }
      VLOG(1) << "Track backward from " << f;
      MotionVectorFrame mvf;
      MotionVectorFrameFromTrackingData(a.chunk_data->item(f).tracking_data(),
                                        &mvf);
      const int64 track_duration_ms =
          TrackingDataDurationMs(a.chunk_data->item(f));
      if (track_duration_ms > 0) {
        mvf.duration_ms = track_duration_ms;
      }
      const bool forward_tracking = false;
      if (!motion_box.TrackStep(f, mvf, forward_tracking)) {
        VLOG(1) << "Failed backward at frame: " << f;
        break;
      } else {
        // Test if current request is canceled.
        {
          absl::MutexLock lock(&status_mutex_);
          if (track_status_[a.id][a.checkpoint].canceled) {
            --track_status_[a.id][a.checkpoint].tracks_ongoing;
            status_condvar_.SignalAll();
            return;
          }
        }

        TimedBox result;
        const MotionBoxState& result_state = motion_box.StateAtFrame(f - 1);
        TimedBoxFromMotionBoxState(result_state, &result);
        result.time_msec = a.chunk_data->item(f).prev_timestamp_usec() / 1000;
        AddBoxResult(result, a.id, a.checkpoint, result_state);
      }

      if (f == first_frame && !a.chunk_data->first_chunk()) {
        VLOG(1) << "Read next chunk: " << f << "==" << first_frame << " in "
                << a.chunk_idx;
        // First frame, successful track, continue.
        AugmentedChunkPtr prev_chunk(
            ReadChunk(a.id, a.checkpoint, a.chunk_idx - 1));
        if (prev_chunk.first != nullptr) {
          const int last_frame = prev_chunk.first->item_size() - 1;
          TrackingImplArgs prev_args(prev_chunk, motion_box.StateAtFrame(f - 1),
                                     last_frame, a.chunk_idx - 1, a.id,
                                     a.checkpoint, a.forward, false, a.min_msec,
                                     a.max_msec);

          TrackingImpl(prev_args);
        } else {
          cleanup_func();
          LOG(ERROR) << "Can't read expected chunk file! " << a.chunk_idx - 1
                     << " while tracking @"
                     << a.chunk_data->item(f).timestamp_usec() / 1000
                     << " with cutoff " << a.min_msec;
          return;
        }
      }
    }
  }

  cleanup_func();
}

bool TimedBoxAtTime(const PathSegment& segment, int64 time_msec, TimedBox* box,
                    MotionBoxState* state) {
  CHECK(box);

  if (segment.empty()) {
    return false;
  }

  TimedBox to_find;
  to_find.time_msec = time_msec;
  auto pos = std::lower_bound(segment.begin(), segment.end(), to_find);
  if (pos != segment.end() && pos->time_msec == time_msec) {
    *box = *pos;
    if (state) {
      *state = *pos->state;
    }
    return true;
  }

  constexpr int kMaxDiff = 67;

  if (pos == segment.begin()) {
    if (pos->time_msec - time_msec < kMaxDiff) {
      *box = *pos;
      if (state && pos->state) {
        *state = *pos->state;
      }
      return true;
    } else {
      return false;
    }
  }

  if (pos == segment.end()) {
    if (time_msec - pos[-1].time_msec < kMaxDiff) {
      *box = pos[-1];
      if (state && pos[-1].state) {
        *state = *pos[-1].state;
      }
      return true;
    } else {
      return false;
    }
  }

  // Interpolation necessary.
  *box = BlendTimedBoxes(pos[-1], pos[0], time_msec);
  if (state) {
    // Grab closest state.
    if (std::abs(pos[-1].time_msec - time_msec) <
        std::abs(pos[0].time_msec - time_msec)) {
      if (pos[-1].state) {
        *state = *pos[-1].state;
      }
    } else {
      if (pos[0].state) {
        *state = *pos[0].state;
      }
    }
  }
  return true;
}

void BoxTracker::ResumeTracking() {
  absl::MutexLock lock(&status_mutex_);
  canceling_ = false;
}

void BoxTracker::CancelAllOngoingTracks() {
  // Get a list of items to be canceled (id, checkpoint)
  absl::MutexLock lock(&status_mutex_);
  canceling_ = true;

  std::vector<std::pair<int, int>> to_be_canceled;
  for (auto& id : track_status_) {
    for (auto& checkpoint : id.second) {
      if (checkpoint.second.tracks_ongoing > 0) {
        checkpoint.second.canceled = true;
        to_be_canceled.push_back(std::make_pair(id.first, checkpoint.first));
      }
    }
  }

  // Wait for ongoing requests to terminate.
  auto on_going_test = [&to_be_canceled, this]() -> bool {
    status_mutex_.AssertHeld();
    for (const auto& item : to_be_canceled) {
      if (track_status_[item.first][item.second].tracks_ongoing > 0) {
        return true;
      }
    }
    return false;
  };

  while (on_going_test()) {
    status_condvar_.Wait(&status_mutex_);
  }

  // Indicate we are done canceling.
  for (const auto& item : to_be_canceled) {
    track_status_[item.first][item.second].canceled = false;
  }
}

bool BoxTracker::WaitForAllOngoingTracks(int timeout_us) {
  MEASURE_TIME << "Tracking time ...";
  absl::MutexLock lock(&status_mutex_);

  // Infinite wait for timeout <= 0.
  absl::Duration timeout = timeout_us > 0 ? absl::Microseconds(timeout_us)
                                          : absl::InfiniteDuration();

  while (timeout > absl::ZeroDuration() && IsTrackingOngoingMutexHeld()) {
    absl::Time start_wait = absl::Now();
    status_condvar_.WaitWithTimeout(&status_mutex_, timeout);

    absl::Duration elapsed = absl::Now() - start_wait;
    timeout -= elapsed;
  }

  return !IsTrackingOngoingMutexHeld();
}

bool BoxTracker::GetTrackingData(int id, int64 request_time_msec,
                                 TrackingData* tracking_data,
                                 int* tracking_data_msec) {
  CHECK(tracking_data);

  int chunk_idx = ChunkIdxFromTime(request_time_msec);

  AugmentedChunkPtr tracking_chunk(ReadChunk(id, kInitCheckpoint, chunk_idx));
  if (!tracking_chunk.first) {
    absl::MutexLock lock(&status_mutex_);
    --track_status_[id][kInitCheckpoint].tracks_ongoing;
    LOG(ERROR) << "Could not read tracking chunk from file.";
    return false;
  }

  std::unique_ptr<const TrackingDataChunk> owned_chunk;
  if (tracking_chunk.second) {
    owned_chunk.reset(tracking_chunk.first);
  }

  const int closest_frame =
      ClosestFrameIndex(request_time_msec, *tracking_chunk.first);

  *tracking_data = tracking_chunk.first->item(closest_frame).tracking_data();
  if (tracking_data_msec) {
    *tracking_data_msec =
        tracking_chunk.first->item(closest_frame).timestamp_usec() / 1000;
  }
  return true;
}

}  // namespace mediapipe
