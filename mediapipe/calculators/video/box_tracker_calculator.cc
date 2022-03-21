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

#include <stdio.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/numbers.h"
#include "mediapipe/calculators/video/box_tracker_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/options_util.h"
#include "mediapipe/util/tracking/box_tracker.h"
#include "mediapipe/util/tracking/tracking.h"
#include "mediapipe/util/tracking/tracking_visualization_utilities.h"

namespace mediapipe {

const char kOptionsTag[] = "OPTIONS";

// A calculator to track box positions over time.
// This calculator works in two modes:
// a) Streaming mode, forward tracking only uses per frame TRACKING TrackingData
//    supplied by tracking. For faster processing use TRACK_TIME to explicitly
//    request tracking results at higher FPS than supplied by TRACKING.
// b) Batch mode: Tracks from tracking chunk files as specified by CACHE_DIR
//    side packet (forward and backward with multiple key framing support).
//    NOTE: When using batch mode you might need some external logic
//    to clear out the caching directory between runs / files; or otherwise
//    stale chunk files might be used.
//
// Initial positions can be either supplied via calculator options or
// INITIAL_POS (not supported on mobile) side packet, but not both.

// Input stream:
//   TRACKING: Input tracking data (proto TrackingData, required if CACHE_DIR
//             is not defined)
//   TRACK_TIME: Timestamps that tracking results should be generated at.
//               Optional. Results generated at a TRACK_TIME w/o corresponding
//               TRACKING packet will be queued up and returned when the next
//               TRACKING input is observed. For those packets also no
//               visualization output will be generated.
//               Can be Packet of any type.
//   VIDEO:    Optional input video stream tracked boxes are rendered over.
//   START:    Optional input stream with PreStream packet to begin processing.
//             Typical use case: When used in batch mode have
//             FlowPackagerCalculator emit a COMPLETE packet to indicate caching
//             is completed.
//  START_POS: Optional initial positions to be tracked as TimedBoxProtoList.
//             Timestamp of the box is used, so box timestamp does not have to
//             be monotonic. Assign monotonic increasing timestamps for
//             START_POS, e.g. 1,2,3 per request.
//             Supplied starting positions are 'fast forwarded', i.e. quickly
//             tracked towards current track head, i.e. last received
//             TrackingData and added to current set of tracked boxes.
//             This is recommended to be used with SyncSetInputStreamHandler.
//  START_POS_PROTO_STRING: Same as START_POS, but is in the form of serialized
//             protobuffer string. When both START_POS and
//             START_POS_PROTO_STRING are present, START_POS is used. Suggest
//             to specify only one of them.
//   RESTART_POS: Same as START_POS, but exclusively for receiving detection
//             results from reacquisition.
//   CANCEL_OBJECT_ID: Optional id of box to be removed. This is recommended
//             to be used with SyncSetInputStreamHandler.
//   RA_TRACK: Performs random access tracking within the specified
//             tracking cache, which is specified in the options of this
//             calculator BoxTrackerCalculatorOptions. Input is of type
//             TimedBoxProtoList.
//             Assumed to be supplied as pair
//             [start0, stop0, start1, stop1, ...] of boxes,
//             (that is list size() % 2 == 0), where position, id and time
//             is used for start, and only time for stop; that is position
//             is ignored.
//             Assign monotonically increasing packet timestamps for RA_TRACK,
//             e.g. 1,2,3; however the timestamp in TimedBoxProtoList
//             can be in arbitrary order.
//             Use with SyncSetInputStreamHandler in streaming mode only.
//   RA_TRACK_PROTO_STRING: Same as RA_TRACK, but is in the form of serialized
//             protobuffer string. When both RA_TRACK and
//             RA_TRACK_PROTO_STRING are present, RA_TRACK is used. Suggest
//             to specify only one of them.
//
// Output streams:
//   VIZ:   Optional output video stream with rendered box positions
//          (requires VIDEO to be present)
//   BOXES: Optional output stream of type TimedBoxProtoList for each
//          initialized result.
//   RA_BOXES: Optional output stream of type TimedBoxProtoList for each
//             request in RA_TRACK. Same timestamp as request is used.
//
// Input side packets:
//   INITIAL_POS: Optional initial positions to be tracked as text format proto
//                of type TimedBoxProtoList. Can not be combined with initial
//                position option. NOT SUPPORTED ON MOBILE.
//   CACHE_DIR:   Optional caching directory tracking chunk files are read
//                from.
//
//
class BoxTrackerCalculator : public CalculatorBase {
 public:
  ~BoxTrackerCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 protected:
  void RenderStates(const std::vector<MotionBoxState>& states, cv::Mat* mat);
  void RenderInternalStates(const std::vector<MotionBoxState>& states,
                            cv::Mat* mat);

  // MotionBox and corresponding PathSegment of results; used in streaming mode.
  struct MotionBoxPath {
    MotionBoxPath(MotionBox&& box_, PathSegment&& path_, bool reacq_ = false)
        : box(std::move(box_)), path(std::move(path_)), reacquisition(reacq_) {}

    MotionBoxPath() = default;

    // Trims the state for forward/backward tracking.
    void Trim(const int cache_size, const bool forward) {
      if (forward) {
        // Trims the box's states queue.
        box.TrimFront(cache_size);
        // Trims the path queue.
        int trim_count = path.size() - cache_size;
        while (trim_count-- > 0) {
          path.pop_front();
        }
      } else {  // backward
        // Trims the box's states queue.
        box.TrimBack(cache_size);
        // Trims the path queue.
        int trim_count = path.size() - cache_size;
        while (trim_count-- > 0) {
          path.pop_back();
        }
      }
    }

    MotionBox box;
    PathSegment path;
    bool reacquisition;
  };

  // MotionBoxPath per unique id that we are tracking.
  typedef absl::node_hash_map<int, MotionBoxPath> MotionBoxMap;

  // Performs tracking of all MotionBoxes in box_map by one frame forward or
  // backward to or from data_frame_num using passed TrackingData.
  // Specify destination timestamp and frame duration TrackingData was
  // computed for. Used in streaming mode.
  // Returns list of ids that failed.
  void StreamTrack(const TrackingData& data, int data_frame_num,
                   int64 dst_timestamp_ms, int64 duration_ms, bool forward,
                   MotionBoxMap* box_map, std::vector<int>* failed_ids);

  // Fast forwards specified boxes from starting position to current play head
  // and outputs successful boxes to box_map.
  // Specify the timestamp boxes are tracked from via timestamp in each
  // TimedBox.
  void FastForwardStartPos(const TimedBoxProtoList& start_pos_list,
                           MotionBoxMap* box_map);

  // Performs random access tracking from box_list (start,stop) tuples and
  // outputs results.
  void OutputRandomAccessTrack(const TimedBoxProtoList& box_list,
                               CalculatorContext* cc);

 private:
  BoxTrackerCalculatorOptions options_;

  TimedBoxProtoList initial_pos_;

  // Keeps tracks boxes that have already been initialized.
  absl::node_hash_set<int> initialized_ids_;

  // Non empty for batch mode tracking.
  std::string cache_dir_;
  // Ids to be tracked in batch_mode.
  absl::node_hash_set<int> batch_track_ids_;

  int frame_num_ = 0;

  // Boxes that are tracked in streaming mode.
  MotionBoxMap streaming_motion_boxes_;

  absl::node_hash_map<int, std::pair<TimedBox, TimedBox>> last_tracked_boxes_;
  int frame_num_since_reset_ = 0;

  // Cache used during streaming mode for fast forward tracking.
  std::deque<std::pair<Timestamp, TrackingData>> tracking_data_cache_;

  // Indicator to track if box_tracker_ has started tracking.
  bool tracking_issued_ = false;
  std::unique_ptr<BoxTracker> box_tracker_;

  // If set, renders tracking data into VIZ stream.
  bool visualize_tracking_data_ = false;

  // If set, renders the box state and internal box state into VIZ stream.
  bool visualize_state_ = false;
  bool visualize_internal_state_ = false;

  // Timestamps for every tracking data input frame.
  std::deque<Timestamp> track_timestamps_;

  // For pruning track_timestamps_ queue.
  static const int kTrackTimestampsMinQueueSize;

  // The long-running index of the head of track_timestamps_.
  int track_timestamps_base_index_ = 0;

  // For pruning MotionBoxPath's state/path queues.
  static const int kMotionBoxPathMinQueueSize;

  // Queued track time requests.
  std::vector<Timestamp> queued_track_requests_;

  // Stores the tracked ids that have been discarded actively, from continuous
  // tracking data. It may accumulate across multiple frames. Once consumed, it
  // should be cleared immediately.
  absl::flat_hash_set<int> actively_discarded_tracked_ids_;

  // Add smooth transition between re-acquisition and previous tracked boxes.
  // `result_box` is the tracking result of one specific timestamp. The smoothed
  // result will be updated in place.
  // `subframe_alpha` is from 0 to 1 (0, 1 repressents previous and current
  // frame with TRACKING_DATA). Any frames with TRACK_TIME should interpolate in
  // between.
  void AddSmoothTransitionToOutputBox(int box_id, TimedBox* result_box,
                                      float subframe_alpha = 1.0f);

  std::deque<Timestamp>::iterator GetRandomAccessTimestampPos(
      const TimedBoxProto& start, bool forward_track);

  std::deque<std::pair<Timestamp, TrackingData>>::iterator
  GetRandomAccessStartData(
      const std::deque<Timestamp>::iterator& timestamp_pos);

  MotionBoxMap PrepareRandomAccessTrack(
      const TimedBoxProto& start, int init_frame, bool forward_track,
      const std::deque<std::pair<Timestamp, TrackingData>>::iterator&
          start_data);

  bool RunForwardTrack(
      const std::deque<std::pair<Timestamp, TrackingData>>::iterator&
          start_data,
      int init_frame, MotionBoxMap* single_map, int64 end_time_msec);

  bool RunBackwardTrack(
      const std::deque<std::pair<Timestamp, TrackingData>>::iterator&
          start_data,
      int init_frame, MotionBoxMap* single_map, int64 end_time_msec);

  void ObtainResultOfRandomAccessTrack(
      const MotionBoxMap& single_map, const TimedBoxProto& start,
      int64 end_time_msec,
      const std::unique_ptr<TimedBoxProtoList>& result_list);
};

REGISTER_CALCULATOR(BoxTrackerCalculator);

// At least 2 timestamps need to be present in track_timestamps_ or streaming
// logic's duration calculation will break.
const int BoxTrackerCalculator::kTrackTimestampsMinQueueSize = 2;

// At least 2: the newly added state, and one from the history.
const int BoxTrackerCalculator::kMotionBoxPathMinQueueSize = 2;

namespace {

constexpr char kCacheDirTag[] = "CACHE_DIR";
constexpr char kInitialPosTag[] = "INITIAL_POS";
constexpr char kRaBoxesTag[] = "RA_BOXES";
constexpr char kBoxesTag[] = "BOXES";
constexpr char kVizTag[] = "VIZ";
constexpr char kRaTrackProtoStringTag[] = "RA_TRACK_PROTO_STRING";
constexpr char kRaTrackTag[] = "RA_TRACK";
constexpr char kCancelObjectIdTag[] = "CANCEL_OBJECT_ID";
constexpr char kRestartPosTag[] = "RESTART_POS";
constexpr char kStartPosProtoStringTag[] = "START_POS_PROTO_STRING";
constexpr char kStartPosTag[] = "START_POS";
constexpr char kStartTag[] = "START";
constexpr char kVideoTag[] = "VIDEO";
constexpr char kTrackTimeTag[] = "TRACK_TIME";
constexpr char kTrackingTag[] = "TRACKING";

// Convert box position according to rotation angle in degrees.
void ConvertCoordinateForRotation(float in_top, float in_left, float in_bottom,
                                  float in_right, int rotation, float* out_top,
                                  float* out_left, float* out_bottom,
                                  float* out_right) {
  CHECK(out_top != nullptr);
  CHECK(out_left != nullptr);
  CHECK(out_bottom != nullptr);
  CHECK(out_right != nullptr);
  const float in_center_x = (in_left + in_right) * 0.5f;
  const float in_center_y = (in_top + in_bottom) * 0.5f;
  const float in_width = in_right - in_left;
  const float in_height = in_bottom - in_top;
  CHECK_GT(in_width, 0);
  CHECK_GT(in_height, 0);
  float out_center_x;
  float out_center_y;
  float out_width;
  float out_height;
  switch (rotation) {
    case 0:
      out_center_x = in_center_x;
      out_center_y = in_center_y;
      out_width = in_width;
      out_height = in_height;
      break;
    case -270:  // FALL_THROUGH_INTENDED
    case 90:
      out_center_x = 1 - in_center_y;
      out_center_y = in_center_x;
      out_width = in_height;
      out_height = in_width;
      break;
    case -180:  // FALL_THROUGH_INTENDED
    case 180:
      out_center_x = 1 - in_center_x;
      out_center_y = 1 - in_center_y;
      out_width = in_width;
      out_height = in_height;
      break;
    case -90:  // FALL_THROUGH_INTENDED
    case 270:
      out_center_x = in_center_y;
      out_center_y = 1 - in_center_x;
      out_width = in_height;
      out_height = in_width;
      break;
    default:
      LOG(ERROR) << "invalid rotation " << rotation;
      out_center_x = in_center_x;
      out_center_y = in_center_y;
      out_width = in_width;
      out_height = in_height;
      break;
  }
  *out_top = out_center_y - out_height * 0.5f;
  *out_left = out_center_x - out_width * 0.5f;
  *out_bottom = out_center_y + out_height * 0.5f;
  *out_right = out_center_x + out_width * 0.5f;
}

void AddStateToPath(const MotionBoxState& state, int64 time_msec,
                    PathSegment* path) {
  CHECK(path);
  TimedBox result;
  TimedBoxFromMotionBoxState(state, &result);
  result.time_msec = time_msec;

  const auto insert_pos = std::lower_bound(path->begin(), path->end(), result);
  // Do not duplicate box positions.
  if (insert_pos == path->end() || insert_pos->time_msec != time_msec) {
    path->insert(insert_pos,
                 InternalTimedBox(result, new MotionBoxState(state)));
  } else {
    LOG(ERROR) << "Box at time " << time_msec << " already present; ignoring";
  }
}

}  // namespace.

absl::Status BoxTrackerCalculator::GetContract(CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kTrackingTag)) {
    cc->Inputs().Tag(kTrackingTag).Set<TrackingData>();
  }

  if (cc->Inputs().HasTag(kTrackTimeTag)) {
    RET_CHECK(cc->Inputs().HasTag(kTrackingTag))
        << "TRACK_TIME needs TRACKING input";
    cc->Inputs().Tag(kTrackTimeTag).SetAny();
  }

  if (cc->Inputs().HasTag(kVideoTag)) {
    cc->Inputs().Tag(kVideoTag).Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag(kStartTag)) {
    // Actual packet content does not matter.
    cc->Inputs().Tag(kStartTag).SetAny();
  }

  if (cc->Inputs().HasTag(kStartPosTag)) {
    cc->Inputs().Tag(kStartPosTag).Set<TimedBoxProtoList>();
  }

  if (cc->Inputs().HasTag(kStartPosProtoStringTag)) {
    cc->Inputs().Tag(kStartPosProtoStringTag).Set<std::string>();
  }

  if (cc->Inputs().HasTag(kRestartPosTag)) {
    cc->Inputs().Tag(kRestartPosTag).Set<TimedBoxProtoList>();
  }

  if (cc->Inputs().HasTag(kCancelObjectIdTag)) {
    cc->Inputs().Tag(kCancelObjectIdTag).Set<int>();
  }

  if (cc->Inputs().HasTag(kRaTrackTag)) {
    cc->Inputs().Tag(kRaTrackTag).Set<TimedBoxProtoList>();
  }

  if (cc->Inputs().HasTag(kRaTrackProtoStringTag)) {
    cc->Inputs().Tag(kRaTrackProtoStringTag).Set<std::string>();
  }

  if (cc->Outputs().HasTag(kVizTag)) {
    RET_CHECK(cc->Inputs().HasTag(kVideoTag))
        << "Output stream VIZ requires VIDEO to be present.";
    cc->Outputs().Tag(kVizTag).Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag(kBoxesTag)) {
    cc->Outputs().Tag(kBoxesTag).Set<TimedBoxProtoList>();
  }

  if (cc->Outputs().HasTag(kRaBoxesTag)) {
    cc->Outputs().Tag(kRaBoxesTag).Set<TimedBoxProtoList>();
  }

#if defined(__ANDROID__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)
  RET_CHECK(!cc->InputSidePackets().HasTag(kInitialPosTag))
      << "Unsupported on mobile";
#else
  if (cc->InputSidePackets().HasTag(kInitialPosTag)) {
    cc->InputSidePackets().Tag(kInitialPosTag).Set<std::string>();
  }
#endif  // defined(__ANDROID__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)

  if (cc->InputSidePackets().HasTag(kCacheDirTag)) {
    cc->InputSidePackets().Tag(kCacheDirTag).Set<std::string>();
  }

  RET_CHECK(cc->Inputs().HasTag(kTrackingTag) !=
            cc->InputSidePackets().HasTag(kCacheDirTag))
      << "Either TRACKING or CACHE_DIR needs to be specified.";

  if (cc->InputSidePackets().HasTag(kOptionsTag)) {
    cc->InputSidePackets().Tag(kOptionsTag).Set<CalculatorOptions>();
  }

  return absl::OkStatus();
}

absl::Status BoxTrackerCalculator::Open(CalculatorContext* cc) {
  options_ = tool::RetrieveOptions(cc->Options<BoxTrackerCalculatorOptions>(),
                                   cc->InputSidePackets(), kOptionsTag);

  RET_CHECK(!cc->InputSidePackets().HasTag(kInitialPosTag) ||
            !options_.has_initial_position())
      << "Can not specify initial position as side packet and via options";

  if (options_.has_initial_position()) {
    initial_pos_ = options_.initial_position();
  }

#if !defined(__ANDROID__) && !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
  if (cc->InputSidePackets().HasTag(kInitialPosTag)) {
    LOG(INFO) << "Parsing: "
              << cc->InputSidePackets().Tag(kInitialPosTag).Get<std::string>();
    initial_pos_ = ParseTextProtoOrDie<TimedBoxProtoList>(
        cc->InputSidePackets().Tag(kInitialPosTag).Get<std::string>());
  }
#endif  // !defined(__ANDROID__) && !defined(__APPLE__) &&
        // !defined(__EMSCRIPTEN__)

  // Compile list of ids to be tracked.
  for (const auto& pos : initial_pos_.box()) {
    RET_CHECK(pos.id() >= 0) << "Requires id to be set";
    batch_track_ids_.insert(pos.id());
  }

  visualize_tracking_data_ =
      options_.visualize_tracking_data() && cc->Outputs().HasTag(kVizTag);
  visualize_state_ =
      options_.visualize_state() && cc->Outputs().HasTag(kVizTag);
  visualize_internal_state_ =
      options_.visualize_internal_state() && cc->Outputs().HasTag(kVizTag);

  // Force recording of internal state for rendering.
  if (visualize_internal_state_) {
    options_.mutable_tracker_options()
        ->mutable_track_step_options()
        ->set_return_internal_state(true);
  }

  if (visualize_state_ || visualize_internal_state_) {
    options_.mutable_tracker_options()->set_record_path_states(true);
  }

  if (cc->InputSidePackets().HasTag(kCacheDirTag)) {
    cache_dir_ = cc->InputSidePackets().Tag(kCacheDirTag).Get<std::string>();
    RET_CHECK(!cache_dir_.empty());
    box_tracker_.reset(new BoxTracker(cache_dir_, options_.tracker_options()));
  } else {
    // Check that all boxes have a unique id.
    RET_CHECK(initial_pos_.box_size() == batch_track_ids_.size())
        << "In streaming mode every box must be given its unique id";
  }

  if (options_.streaming_track_data_cache_size() > 0) {
    RET_CHECK(!cc->InputSidePackets().HasTag(kCacheDirTag))
        << "Streaming mode not compatible with cache dir.";
  }

  return absl::OkStatus();
}

absl::Status BoxTrackerCalculator::Process(CalculatorContext* cc) {
  // Batch mode, issue tracking requests.
  if (box_tracker_ && !tracking_issued_) {
    for (const auto& pos : initial_pos_.box()) {
      box_tracker_->NewBoxTrack(TimedBox::FromProto(pos), pos.id());
    }
    tracking_issued_ = true;
  }

  const Timestamp& timestamp = cc->InputTimestamp();
  if (timestamp == Timestamp::PreStream()) {
    // Indicator packet.
    return absl::OkStatus();
  }

  InputStream* track_stream = cc->Inputs().HasTag(kTrackingTag)
                                  ? &(cc->Inputs().Tag(kTrackingTag))
                                  : nullptr;
  InputStream* track_time_stream = cc->Inputs().HasTag(kTrackTimeTag)
                                       ? &(cc->Inputs().Tag(kTrackTimeTag))
                                       : nullptr;

  // Cache tracking data if possible.
  if (track_stream && !track_stream->IsEmpty()) {
    const TrackingData& track_data = track_stream->Get<TrackingData>();
    const int track_cache_size = options_.streaming_track_data_cache_size();
    if (track_cache_size > 0) {
      tracking_data_cache_.push_back(std::make_pair(timestamp, track_data));
      while (tracking_data_cache_.size() > track_cache_size) {
        tracking_data_cache_.pop_front();
      }
    }
    track_timestamps_.push_back(timestamp);
    int trim_count = track_timestamps_.size() -
                     std::max(track_cache_size, kTrackTimestampsMinQueueSize);
    if (trim_count > 0) {
      track_timestamps_base_index_ += trim_count;

      while (trim_count-- > 0) {
        track_timestamps_.pop_front();
      }
    }
  }

  InputStream* start_pos_stream = cc->Inputs().HasTag(kStartPosTag)
                                      ? &(cc->Inputs().Tag(kStartPosTag))
                                      : nullptr;

  MotionBoxMap fast_forward_boxes;
  if (start_pos_stream && !start_pos_stream->IsEmpty()) {
    // Try to fast forward boxes to current tracking head.
    const TimedBoxProtoList& start_pos_list =
        start_pos_stream->Get<TimedBoxProtoList>();
    FastForwardStartPos(start_pos_list, &fast_forward_boxes);
  }

  InputStream* start_pos_proto_string_stream =
      cc->Inputs().HasTag(kStartPosProtoStringTag)
          ? &(cc->Inputs().Tag(kStartPosProtoStringTag))
          : nullptr;
  if (start_pos_stream == nullptr || start_pos_stream->IsEmpty()) {
    if (start_pos_proto_string_stream &&
        !start_pos_proto_string_stream->IsEmpty()) {
      auto start_pos_list_str =
          start_pos_proto_string_stream->Get<std::string>();
      TimedBoxProtoList start_pos_list;
      start_pos_list.ParseFromString(start_pos_list_str);
      FastForwardStartPos(start_pos_list, &fast_forward_boxes);
    }
  }

  InputStream* restart_pos_stream = cc->Inputs().HasTag(kRestartPosTag)
                                        ? &(cc->Inputs().Tag(kRestartPosTag))
                                        : nullptr;

  if (restart_pos_stream && !restart_pos_stream->IsEmpty()) {
    const TimedBoxProtoList& restart_pos_list =
        restart_pos_stream->Get<TimedBoxProtoList>();
    FastForwardStartPos(restart_pos_list, &fast_forward_boxes);
  }

  InputStream* cancel_object_id_stream =
      cc->Inputs().HasTag(kCancelObjectIdTag)
          ? &(cc->Inputs().Tag(kCancelObjectIdTag))
          : nullptr;
  if (cancel_object_id_stream && !cancel_object_id_stream->IsEmpty()) {
    const int cancel_object_id = cancel_object_id_stream->Get<int>();
    if (streaming_motion_boxes_.erase(cancel_object_id) == 0) {
      LOG(WARNING) << "box id " << cancel_object_id << " does not exist.";
    }
  }

  cv::Mat input_view;
  cv::Mat viz_view;
  std::unique_ptr<ImageFrame> viz_frame;

  TrackingData track_data_to_render;

  if (cc->Outputs().HasTag(kVizTag)) {
    InputStream* video_stream = &(cc->Inputs().Tag(kVideoTag));
    if (!video_stream->IsEmpty()) {
      input_view = formats::MatView(&video_stream->Get<ImageFrame>());

      viz_frame.reset(new ImageFrame());
      viz_frame->CopyFrom(video_stream->Get<ImageFrame>(), 16);
      viz_view = formats::MatView(viz_frame.get());
    }
  }

  // Results to be output or rendered, list of TimedBox for every id that are
  // present at this frame.
  TimedBoxProtoList box_track_list;

  CHECK(box_tracker_ || track_stream)
      << "Expected either batch or streaming mode";

  // Corresponding list of box states for rendering. For each id present at
  // this frame stores closest 1-2 states.
  std::vector<std::vector<MotionBoxState>> box_state_list;
  int64 timestamp_msec = timestamp.Value() / 1000;

  if (box_tracker_) {  // Batch mode.
    // Ensure tracking has terminated.
    box_tracker_->WaitForAllOngoingTracks();

    // Cycle through ids.
    for (int id : batch_track_ids_) {
      TimedBox result;
      std::vector<MotionBoxState> states;
      std::vector<MotionBoxState>* states_ptr =
          (visualize_state_ || visualize_internal_state_) ? &states : nullptr;

      if (box_tracker_->GetTimedPosition(id, timestamp_msec, &result,
                                         states_ptr)) {
        TimedBoxProto proto = result.ToProto();
        proto.set_id(id);
        *box_track_list.add_box() = std::move(proto);

        if (states_ptr) {
          box_state_list.push_back(*states_ptr);
        }
      }
    }

    if (visualize_tracking_data_) {
      constexpr int kVizId = -1;
      box_tracker_->GetTrackingData(kVizId, timestamp_msec,
                                    &track_data_to_render);
    }
  } else {
    // Streaming mode.
    // If track data is available advance all boxes by new data.
    if (!track_stream->IsEmpty()) {
      const TrackingData& track_data = track_stream->Get<TrackingData>();

      if (visualize_tracking_data_) {
        track_data_to_render = track_data;
      }

      const int64 time_ms = track_timestamps_.back().Value() / 1000;
      const int64 duration_ms =
          track_timestamps_.size() > 1
              ? time_ms - track_timestamps_.rbegin()[1].Value() / 1000
              : 0;

      std::vector<int> failed_boxes;
      StreamTrack(track_data, frame_num_, time_ms, duration_ms,
                  true,  // forward.
                  &streaming_motion_boxes_, &failed_boxes);

      // Add fast forward boxes.
      if (!fast_forward_boxes.empty()) {
        for (const auto& box : fast_forward_boxes) {
          streaming_motion_boxes_.emplace(box.first, box.second);
        }
        fast_forward_boxes.clear();
      }

      // Remove failed boxes.
      for (int id : failed_boxes) {
        streaming_motion_boxes_.erase(id);
      }

      // Init new boxes once data from previous time to current is available.
      for (const auto& pos : initial_pos_.box()) {
        if (timestamp_msec - pos.time_msec() >= 0 &&
            initialized_ids_.find(pos.id()) == initialized_ids_.end()) {
          MotionBoxState init_state;
          MotionBoxStateFromTimedBox(TimedBox::FromProto(pos), &init_state);

          InitializeInliersOutliersInMotionBoxState(track_data, &init_state);
          InitializePnpHomographyInMotionBoxState(
              track_data, options_.tracker_options().track_step_options(),
              &init_state);

          TrackStepOptions track_step_options =
              options_.tracker_options().track_step_options();
          ChangeTrackingDegreesBasedOnStartPos(pos, &track_step_options);
          MotionBox init_box(track_step_options);

          // Init at previous frame.
          init_box.ResetAtFrame(frame_num_, init_state);

          PathSegment init_path;
          AddStateToPath(init_state, timestamp_msec, &init_path);

          streaming_motion_boxes_.emplace(
              pos.id(), MotionBoxPath(std::move(init_box), std::move(init_path),
                                      pos.reacquisition()));
          initialized_ids_.insert(pos.id());
        }
      }

      ++frame_num_;
    } else {
      // Track stream is empty, if anything is requested on track_time_stream
      // queue up requests.
      if (track_time_stream && !track_time_stream->IsEmpty()) {
        queued_track_requests_.push_back(timestamp);
      }
    }

    // Can output be generated?
    if (!track_stream->IsEmpty()) {
      ++frame_num_since_reset_;

      // Generate results for queued up request.
      if (cc->Outputs().HasTag(kBoxesTag) && !queued_track_requests_.empty()) {
        for (int j = 0; j < queued_track_requests_.size(); ++j) {
          const Timestamp& past_time = queued_track_requests_[j];
          RET_CHECK(past_time.Value() < timestamp.Value())
              << "Inconsistency, queued up requests should occur in past";
          std::unique_ptr<TimedBoxProtoList> past_box_list(
              new TimedBoxProtoList());

          for (auto& motion_box_path : streaming_motion_boxes_) {
            TimedBox result_box;
            TimedBoxAtTime(motion_box_path.second.path,
                           past_time.Value() / 1000, &result_box, nullptr);

            const float subframe_alpha =
                static_cast<float>(j + 1) / (queued_track_requests_.size() + 1);
            AddSmoothTransitionToOutputBox(motion_box_path.first, &result_box,
                                           subframe_alpha);

            TimedBoxProto proto = result_box.ToProto();
            proto.set_id(motion_box_path.first);
            proto.set_reacquisition(motion_box_path.second.reacquisition);
            *past_box_list->add_box() = std::move(proto);
          }

          // Output for every time.
          cc->Outputs().Tag(kBoxesTag).Add(past_box_list.release(), past_time);
        }

        queued_track_requests_.clear();
      }

      // Generate result at current frame.
      for (auto& motion_box_path : streaming_motion_boxes_) {
        TimedBox result_box;
        MotionBoxState result_state;
        TimedBoxAtTime(motion_box_path.second.path, timestamp_msec, &result_box,
                       &result_state);

        AddSmoothTransitionToOutputBox(motion_box_path.first, &result_box);

        TimedBoxProto proto = result_box.ToProto();
        proto.set_id(motion_box_path.first);
        proto.set_reacquisition(motion_box_path.second.reacquisition);
        *box_track_list.add_box() = std::move(proto);

        if (visualize_state_ || visualize_internal_state_) {
          box_state_list.push_back({result_state});
        }
      }
    }
    // end streaming mode case.
  }

  // Save a snapshot of latest tracking results before override with fast
  // forwarded start pos.
  if (!fast_forward_boxes.empty()) {
    frame_num_since_reset_ = 0;
    last_tracked_boxes_.clear();
    // Add any remaining fast forward boxes. For example occurs if START_POS is
    // specified with non-matching TRACKING mode
    for (const auto& reset_box : fast_forward_boxes) {
      const auto tracked_box_iter =
          streaming_motion_boxes_.find(reset_box.first);
      if (tracked_box_iter != streaming_motion_boxes_.end()) {
        if (!reset_box.second.path.empty() &&
            !tracked_box_iter->second.path.empty()) {
          last_tracked_boxes_[reset_box.first] =
              std::make_pair(tracked_box_iter->second.path.back(),
                             reset_box.second.path.back());
        }
      }

      // Override previous tracking with reset start pos.
      streaming_motion_boxes_[reset_box.first] = reset_box.second;
    }
  }

  if (viz_frame) {
    if (visualize_tracking_data_) {
      RenderTrackingData(track_data_to_render, &viz_view);
    }

    if (visualize_state_) {
      for (const auto& state_vec : box_state_list) {
        RenderStates(state_vec, &viz_view);
      }
    }

    if (visualize_internal_state_) {
      for (const auto& state_vec : box_state_list) {
        RenderInternalStates(state_vec, &viz_view);
      }
    }

    for (const auto& box : box_track_list.box()) {
      RenderBox(box, &viz_view);
    }
  }

  // Handle random access track requests.
  InputStream* ra_track_stream = cc->Inputs().HasTag(kRaTrackTag)
                                     ? &(cc->Inputs().Tag(kRaTrackTag))
                                     : nullptr;

  if (ra_track_stream && !ra_track_stream->IsEmpty()) {
    RET_CHECK(!box_tracker_) << "Random access only for streaming mode "
                             << "implemented.";
    const TimedBoxProtoList& box_list =
        ra_track_stream->Get<TimedBoxProtoList>();
    RET_CHECK(box_list.box_size() % 2 == 0)
        << "Expect even number of (start,end) tuples but get "
        << box_list.box_size();
    OutputRandomAccessTrack(box_list, cc);
  }

  InputStream* ra_track_proto_string_stream =
      cc->Inputs().HasTag(kRaTrackProtoStringTag)
          ? &(cc->Inputs().Tag(kRaTrackProtoStringTag))
          : nullptr;
  if (ra_track_stream == nullptr || ra_track_stream->IsEmpty()) {
    if (ra_track_proto_string_stream &&
        !ra_track_proto_string_stream->IsEmpty()) {
      RET_CHECK(!box_tracker_) << "Random access only for streaming mode "
                               << "implemented.";
      auto box_list_str = ra_track_proto_string_stream->Get<std::string>();
      TimedBoxProtoList box_list;
      box_list.ParseFromString(box_list_str);
      RET_CHECK(box_list.box_size() % 2 == 0)
          << "Expect even number of (start,end) tuples but get "
          << box_list.box_size();
      OutputRandomAccessTrack(box_list, cc);
    }
  }

  // Always output in batch, only output in streaming if tracking data
  // is present (might be in fast forward mode instead).
  if (cc->Outputs().HasTag(kBoxesTag) &&
      (box_tracker_ || !track_stream->IsEmpty())) {
    std::unique_ptr<TimedBoxProtoList> boxes(new TimedBoxProtoList());
    *boxes = std::move(box_track_list);
    cc->Outputs().Tag(kBoxesTag).Add(boxes.release(), timestamp);
  }

  if (viz_frame) {
    cc->Outputs().Tag(kVizTag).Add(viz_frame.release(), timestamp);
  }

  return absl::OkStatus();
}

void BoxTrackerCalculator::AddSmoothTransitionToOutputBox(
    int box_id, TimedBox* result_box, float subframe_alpha) {
  if (options_.start_pos_transition_frames() > 0 &&
      frame_num_since_reset_ <= options_.start_pos_transition_frames()) {
    const auto& box_iter = last_tracked_boxes_.find(box_id);
    if (box_iter != last_tracked_boxes_.end()) {
      // We first compute the blend of last tracked box with reset box at the
      // same timestamp as blend_start = alpha * reset_box + (1 - alpha) *
      // last_tracked_box. Then apply the motion from current tracking to reset
      // pos to the blended start pos as: result_box = blend_start +
      // (current_box - reset_box) With some derivation, we can get result_box =
      // (1 - alpha) * (last_track - reset_box) + current_box
      TimedBox tmp_box = TimedBox::Blend(box_iter->second.first,
                                         box_iter->second.second, 1.0, -1.0);
      const float alpha = (frame_num_since_reset_ - 1.0f + subframe_alpha) /
                          options_.start_pos_transition_frames();
      *result_box = TimedBox::Blend(tmp_box, *result_box, 1.0 - alpha, 1.0);
    }
  }
}

void BoxTrackerCalculator::OutputRandomAccessTrack(
    const TimedBoxProtoList& box_list, CalculatorContext* cc) {
  std::unique_ptr<TimedBoxProtoList> result_list(new TimedBoxProtoList());

  for (int i = 0; i < box_list.box_size(); i += 2) {
    const TimedBoxProto start = box_list.box(i);
    int64 end_time_msec = box_list.box(i + 1).time_msec();
    const bool forward_track = start.time_msec() < end_time_msec;

    if (track_timestamps_.empty()) {
      LOG(WARNING) << "No tracking data cached yet.";
      continue;
    }

    // Performing the range check in msec (b/138399787)
    const int64 tracking_start_timestamp_msec =
        track_timestamps_.front().Microseconds() / 1000;
    const int64 tracking_end_timestamp_msec =
        track_timestamps_.back().Microseconds() / 1000;
    if (start.time_msec() < tracking_start_timestamp_msec) {
      LOG(WARNING) << "Request start timestamp " << start.time_msec()
                   << " too old. First frame in the window: "
                   << tracking_start_timestamp_msec;
      continue;
    }
    if (start.time_msec() > tracking_end_timestamp_msec) {
      LOG(WARNING) << "Request start timestamp " << start.time_msec()
                   << " too new. Last frame in the window: "
                   << tracking_end_timestamp_msec;
      continue;
    }
    if (end_time_msec < tracking_start_timestamp_msec) {
      LOG(WARNING) << "Request end timestamp " << end_time_msec
                   << " too old. First frame in the window: "
                   << tracking_start_timestamp_msec;
      continue;
    }
    if (end_time_msec > tracking_end_timestamp_msec) {
      LOG(WARNING) << "Request end timestamp " << end_time_msec
                   << " too new. Last frame in the window: "
                   << tracking_end_timestamp_msec;
      continue;
    }

    std::deque<Timestamp>::iterator timestamp_pos =
        GetRandomAccessTimestampPos(start, forward_track);

    if (timestamp_pos == track_timestamps_.end()) {
      LOG(ERROR) << "Random access outside cached range";
      continue;
    }

    // Locate start of tracking data.
    std::deque<std::pair<Timestamp, TrackingData>>::iterator start_data =
        GetRandomAccessStartData(timestamp_pos);

    // TODO: Interpolate random access tracking start_data instead
    // of dropping the request in the case of missing processed frame.
    if (start_data == tracking_data_cache_.end()) {
      LOG(ERROR) << "Random access starts at unprocessed frame.";
      continue;
    }

    const int init_frame = timestamp_pos - track_timestamps_.begin() +
                           track_timestamps_base_index_;
    CHECK_GE(init_frame, 0);

    MotionBoxMap single_map =
        PrepareRandomAccessTrack(start, init_frame, forward_track, start_data);
    bool track_error = forward_track
                           ? RunForwardTrack(start_data, init_frame,
                                             &single_map, end_time_msec)
                           : RunBackwardTrack(start_data, init_frame,
                                              &single_map, end_time_msec);

    if (track_error) {
      LOG(ERROR) << "Could not track box.";
      continue;
    }

    ObtainResultOfRandomAccessTrack(single_map, start, end_time_msec,
                                    result_list);
  }

  cc->Outputs()
      .Tag(kRaBoxesTag)
      .Add(result_list.release(), cc->InputTimestamp());
}

std::deque<Timestamp>::iterator
BoxTrackerCalculator::GetRandomAccessTimestampPos(const TimedBoxProto& start,
                                                  bool forward_track) {
  std::deque<Timestamp>::iterator timestamp_pos;
  Timestamp timestamp(start.time_msec() * 1000);
  if (forward_track) {
    timestamp_pos = std::upper_bound(track_timestamps_.begin(),
                                     track_timestamps_.end(), timestamp);
  } else {
    timestamp_pos = std::lower_bound(track_timestamps_.begin(),
                                     track_timestamps_.end(), timestamp);
  }
  return timestamp_pos;
}

std::deque<std::pair<Timestamp, TrackingData>>::iterator
BoxTrackerCalculator::GetRandomAccessStartData(
    const std::deque<Timestamp>::iterator& timestamp_pos) {
  std::deque<std::pair<Timestamp, TrackingData>>::iterator start_data =
      std::find_if(tracking_data_cache_.begin(), tracking_data_cache_.end(),
                   [timestamp_pos](
                       const std::pair<Timestamp, TrackingData>& item) -> bool {
                     return item.first == *timestamp_pos;
                   });
  return start_data;
}

BoxTrackerCalculator::MotionBoxMap
BoxTrackerCalculator::PrepareRandomAccessTrack(
    const TimedBoxProto& start, int init_frame, bool forward_track,
    const std::deque<std::pair<Timestamp, TrackingData>>::iterator&
        start_data) {
  MotionBoxMap single_map;
  // Init state at request time.
  MotionBoxState init_state;
  MotionBoxStateFromTimedBox(TimedBox::FromProto(start), &init_state);

  InitializeInliersOutliersInMotionBoxState(start_data->second, &init_state);
  InitializePnpHomographyInMotionBoxState(
      start_data->second, options_.tracker_options().track_step_options(),
      &init_state);

  TrackStepOptions track_step_options =
      options_.tracker_options().track_step_options();
  ChangeTrackingDegreesBasedOnStartPos(start, &track_step_options);
  MotionBox init_box(track_step_options);
  init_box.ResetAtFrame(init_frame - (forward_track ? 1 : 0), init_state);

  PathSegment init_path;

  // Avoid duplicating start time in case TrackingData has same value.
  // Note: For backward tracking we always arrive at an earlier frame, so
  // no duplication can happen, see StreamTrack for details.
  if (start.time_msec() != start_data->first.Value() / 1000 || !forward_track) {
    AddStateToPath(init_state, start.time_msec(), &init_path);
  }

  single_map.emplace(start.id(),
                     MotionBoxPath(std::move(init_box), std::move(init_path)));
  return single_map;
}

bool BoxTrackerCalculator::RunForwardTrack(
    const std::deque<std::pair<Timestamp, TrackingData>>::iterator& start_data,
    int init_frame, MotionBoxMap* single_map, int64 end_time_msec) {
  int curr_frame = init_frame;
  for (auto cache_pos = start_data; cache_pos != tracking_data_cache_.end();
       ++cache_pos, ++curr_frame) {
    std::vector<int> failed_box;
    const int64 dst_time_msec = cache_pos->first.Value() / 1000;
    const int64 curr_duration =
        (cache_pos == tracking_data_cache_.begin())
            ? 0
            : (cache_pos[0].first.Value() - cache_pos[-1].first.Value()) / 1000;
    StreamTrack(cache_pos->second, curr_frame, dst_time_msec, curr_duration,
                true,  // forward
                single_map, &failed_box);
    if (!failed_box.empty()) {
      return true;
    }
    if (dst_time_msec > end_time_msec) {
      return false;
    }
  }
  return false;
}

bool BoxTrackerCalculator::RunBackwardTrack(
    const std::deque<std::pair<Timestamp, TrackingData>>::iterator& start_data,
    int init_frame, MotionBoxMap* single_map, int64 end_time_msec) {
  int curr_frame = init_frame;
  for (auto cache_pos = start_data; cache_pos != tracking_data_cache_.begin();
       --cache_pos, --curr_frame) {
    std::vector<int> failed_box;
    const int64 dst_time_msec = cache_pos[-1].first.Value() / 1000;
    const int64 curr_duration =
        (cache_pos[0].first.Value() - cache_pos[-1].first.Value()) / 1000;
    StreamTrack(cache_pos->second, curr_frame, dst_time_msec, curr_duration,
                false,  // backward
                single_map, &failed_box);
    if (!failed_box.empty()) {
      return true;
    }
    if (dst_time_msec < end_time_msec) {
      return false;
    }
  }
  return false;
}

void BoxTrackerCalculator::ObtainResultOfRandomAccessTrack(
    const MotionBoxMap& single_map, const TimedBoxProto& start,
    int64 end_time_msec,
    const std::unique_ptr<TimedBoxProtoList>& result_list) {
  const MotionBoxPath& result_path = single_map.find(start.id())->second;
  TimedBox result_box;
  TimedBoxAtTime(result_path.path, end_time_msec, &result_box, nullptr);
  TimedBoxProto proto = result_box.ToProto();
  proto.set_id(start.id());
  *result_list->add_box() = std::move(proto);
}

void BoxTrackerCalculator::RenderStates(
    const std::vector<MotionBoxState>& states, cv::Mat* mat) {
  for (int k = 0; k < states.size(); ++k) {
    const bool print_stats = k == 0;
    RenderState(states[k], print_stats, mat);
  }
}

void BoxTrackerCalculator::RenderInternalStates(
    const std::vector<MotionBoxState>& states, cv::Mat* mat) {
  for (const MotionBoxState& state : states) {
    RenderInternalState(state.internal(), mat);
  }
}

void BoxTrackerCalculator::StreamTrack(const TrackingData& data,
                                       int data_frame_num,
                                       int64 dst_timestamp_ms,
                                       int64 duration_ms, bool forward,
                                       MotionBoxMap* box_map,
                                       std::vector<int>* failed_ids) {
  CHECK(box_map);
  CHECK(failed_ids);

  // Cache the actively discarded tracked ids from the new tracking data.
  for (const int discarded_id :
       data.motion_data().actively_discarded_tracked_ids()) {
    actively_discarded_tracked_ids_.insert(discarded_id);
  }

  // Track all existing boxes by one frame.
  MotionVectorFrame mvf;  // Holds motion from current to previous frame.
  MotionVectorFrameFromTrackingData(data, &mvf);
  mvf.actively_discarded_tracked_ids = &actively_discarded_tracked_ids_;

  if (forward) {
    MotionVectorFrame mvf_inverted;
    InvertMotionVectorFrame(mvf, &mvf_inverted);
    std::swap(mvf, mvf_inverted);
  }

  if (duration_ms > 0) {
    mvf.duration_ms = duration_ms;
  }

  const int from_frame = data_frame_num - (forward ? 1 : 0);
  const int to_frame = forward ? from_frame + 1 : from_frame - 1;

  for (auto& motion_box : *box_map) {
    if (!motion_box.second.box.TrackStep(from_frame,  // from frame.
                                         mvf, forward)) {
      failed_ids->push_back(motion_box.first);
      LOG(INFO) << "lost track. pushed failed id: " << motion_box.first;
    } else {
      // Store result.
      PathSegment& path = motion_box.second.path;
      const MotionBoxState& result_state =
          motion_box.second.box.StateAtFrame(to_frame);
      AddStateToPath(result_state, dst_timestamp_ms, &path);
      // motion_box has got new tracking state/path. Now trimming it.
      const int cache_size =
          std::max(options_.streaming_track_data_cache_size(),
                   kMotionBoxPathMinQueueSize);
      motion_box.second.Trim(cache_size, forward);
    }
  }
}

void BoxTrackerCalculator::FastForwardStartPos(
    const TimedBoxProtoList& start_pos_list, MotionBoxMap* box_map) {
  for (const TimedBoxProto& start_pos : start_pos_list.box()) {
    Timestamp timestamp(start_pos.time_msec() * 1000);
    // Locate corresponding frame number for starting position. As TrackingData
    // stores motion from current to last frame; we are using the data after
    // this frame for tracking.
    auto timestamp_pos = std::lower_bound(track_timestamps_.begin(),
                                          track_timestamps_.end(), timestamp);

    if (timestamp_pos == track_timestamps_.end()) {
      LOG(WARNING) << "Received start pos beyond current timestamp, "
                   << "Starting to track once frame arrives.";
      *initial_pos_.add_box() = start_pos;
      continue;
    }

    // Start at previous frame.
    const int init_frame = timestamp_pos - track_timestamps_.begin() +
                           track_timestamps_base_index_;
    CHECK_GE(init_frame, 0);

    // Locate corresponding tracking data.
    auto start_data = std::find_if(
        tracking_data_cache_.begin(), tracking_data_cache_.end(),
        [timestamp_pos](const std::pair<Timestamp, TrackingData>& item)
            -> bool { return item.first == timestamp_pos[0]; });

    if (start_data == tracking_data_cache_.end()) {
      LOG(ERROR) << "Box to fast forward outside tracking data cache. Ignoring."
                 << " To avoid this error consider increasing the cache size.";
      continue;
    }

    // Init state at request time.
    MotionBoxState init_state;
    MotionBoxStateFromTimedBox(TimedBox::FromProto(start_pos), &init_state);

    InitializeInliersOutliersInMotionBoxState(start_data->second, &init_state);
    InitializePnpHomographyInMotionBoxState(
        start_data->second, options_.tracker_options().track_step_options(),
        &init_state);

    TrackStepOptions track_step_options =
        options_.tracker_options().track_step_options();
    ChangeTrackingDegreesBasedOnStartPos(start_pos, &track_step_options);
    MotionBox init_box(track_step_options);
    init_box.ResetAtFrame(init_frame, init_state);

    int curr_frame = init_frame + 1;
    MotionBoxMap single_map;
    PathSegment init_path;
    AddStateToPath(init_state, timestamp_pos[0].Value() / 1000, &init_path);
    single_map.emplace(start_pos.id(),
                       MotionBoxPath(std::move(init_box), std::move(init_path),
                                     start_pos.reacquisition()));
    bool track_error = false;

    for (auto cache_pos = start_data + 1;
         cache_pos != tracking_data_cache_.end(); ++cache_pos, ++curr_frame) {
      std::vector<int> failed_box;
      const int64 curr_time_msec = cache_pos->first.Value() / 1000;
      const int64 curr_duration =
          (cache_pos[0].first.Value() - cache_pos[-1].first.Value()) / 1000;
      StreamTrack(cache_pos->second, curr_frame, curr_time_msec, curr_duration,
                  true,  // forward
                  &single_map, &failed_box);
      if (!failed_box.empty()) {
        LOG(WARNING) << "Unable to fast forward box at frame " << curr_frame;
        track_error = true;
        break;
      }
    }

    if (!track_error) {
      // Fast forward successful.
      if (box_map->find(start_pos.id()) != box_map->end()) {
        DLOG(ERROR) << "Fast forward successful, but box with same id "
                    << "exists already.";
      } else {
        // Add to set of currently tracked boxes.
        const MotionBoxPath& result = single_map.find(start_pos.id())->second;
        box_map->emplace(start_pos.id(), result);
      }
    }
  }
}

}  // namespace mediapipe
