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

#include <fstream>
#include <memory>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/video/flow_packager_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/util/tracking/camera_motion.pb.h"
#include "mediapipe/util/tracking/flow_packager.h"
#include "mediapipe/util/tracking/region_flow.pb.h"

namespace mediapipe {

constexpr char kCacheDirTag[] = "CACHE_DIR";
constexpr char kCompleteTag[] = "COMPLETE";
constexpr char kTrackingChunkTag[] = "TRACKING_CHUNK";
constexpr char kTrackingTag[] = "TRACKING";
constexpr char kCameraTag[] = "CAMERA";
constexpr char kFlowTag[] = "FLOW";

using mediapipe::CameraMotion;
using mediapipe::FlowPackager;
using mediapipe::RegionFlowFeatureList;
using mediapipe::TrackingData;
using mediapipe::TrackingDataChunk;

// A calculator that packages input CameraMotion and RegionFlowFeatureList
// into a TrackingData and optionally writes TrackingDataChunks to file.
//
// Input stream:
//   FLOW:       Input region flow (proto RegionFlowFeatureList).
//   CAMERA:     Input camera stream (proto CameraMotion, optional).
//
// Input side packets:
//   CACHE_DIR:  Optional caching directory tracking files are written to.
//
// Output streams.
//   TRACKING:       Output tracking data (proto TrackingData, per frame
//                   optional).
//   TRACKING_CHUNK: Output tracking chunks (proto TrackingDataChunk,
//                   per chunk, optional), output at the first timestamp
//                   of each chunk.
//   COMPLETE:       Optional output packet sent on PreStream to
//                   to signal downstream calculators that all data has been
//                   processed and calculator is closed. Can be used to indicate
//                   that all data as been written to CACHE_DIR.
class FlowPackagerCalculator : public CalculatorBase {
 public:
  ~FlowPackagerCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

  // Writes passed chunk to disk.
  void WriteChunk(const TrackingDataChunk& chunk) const;

  // Initializes next chunk for tracking beginning from last frame of
  // current chunk (Chunking is design with one frame overlap).
  void PrepareCurrentForNextChunk(TrackingDataChunk* chunk);

 private:
  FlowPackagerCalculatorOptions options_;

  // Caching options.
  bool use_caching_ = false;
  bool build_chunk_ = false;
  std::string cache_dir_;
  int chunk_idx_ = -1;
  TrackingDataChunk tracking_chunk_;

  int frame_idx_ = 0;

  Timestamp prev_timestamp_;
  std::unique_ptr<FlowPackager> flow_packager_;
};

REGISTER_CALCULATOR(FlowPackagerCalculator);

absl::Status FlowPackagerCalculator::GetContract(CalculatorContract* cc) {
  if (!cc->Inputs().HasTag(kFlowTag)) {
    return tool::StatusFail("No input flow was specified.");
  }

  cc->Inputs().Tag(kFlowTag).Set<RegionFlowFeatureList>();

  if (cc->Inputs().HasTag(kCameraTag)) {
    cc->Inputs().Tag(kCameraTag).Set<CameraMotion>();
  }
  if (cc->Outputs().HasTag(kTrackingTag)) {
    cc->Outputs().Tag(kTrackingTag).Set<TrackingData>();
  }
  if (cc->Outputs().HasTag(kTrackingChunkTag)) {
    cc->Outputs().Tag(kTrackingChunkTag).Set<TrackingDataChunk>();
  }
  if (cc->Outputs().HasTag(kCompleteTag)) {
    cc->Outputs().Tag(kCompleteTag).Set<bool>();
  }

  if (cc->InputSidePackets().HasTag(kCacheDirTag)) {
    cc->InputSidePackets().Tag(kCacheDirTag).Set<std::string>();
  }

  return absl::OkStatus();
}

absl::Status FlowPackagerCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<FlowPackagerCalculatorOptions>();

  flow_packager_.reset(new FlowPackager(options_.flow_packager_options()));

  use_caching_ = cc->InputSidePackets().HasTag(kCacheDirTag);
  build_chunk_ = use_caching_ || cc->Outputs().HasTag(kTrackingChunkTag);
  if (use_caching_) {
    cache_dir_ = cc->InputSidePackets().Tag(kCacheDirTag).Get<std::string>();
  }

  return absl::OkStatus();
}

absl::Status FlowPackagerCalculator::Process(CalculatorContext* cc) {
  InputStream* flow_stream = &(cc->Inputs().Tag(kFlowTag));
  const RegionFlowFeatureList& flow = flow_stream->Get<RegionFlowFeatureList>();

  const Timestamp timestamp = flow_stream->Value().Timestamp();

  const CameraMotion* camera_motion = nullptr;
  if (cc->Inputs().HasTag(kCameraTag)) {
    InputStream* camera_stream = &(cc->Inputs().Tag(kCameraTag));
    camera_motion = &camera_stream->Get<CameraMotion>();
  }

  std::unique_ptr<TrackingData> tracking_data(new TrackingData());

  flow_packager_->PackFlow(flow, camera_motion, tracking_data.get());

  if (build_chunk_) {
    if (chunk_idx_ < 0) {  // Lazy init, determine first start.
      chunk_idx_ =
          timestamp.Value() / 1000 / options_.caching_chunk_size_msec();
      tracking_chunk_.set_first_chunk(true);
    }
    CHECK_GE(chunk_idx_, 0);

    TrackingDataChunk::Item* item = tracking_chunk_.add_item();
    item->set_frame_idx(frame_idx_);
    item->set_timestamp_usec(timestamp.Value());
    if (frame_idx_ > 0) {
      item->set_prev_timestamp_usec(prev_timestamp_.Value());
    }
    if (cc->Outputs().HasTag(kTrackingTag)) {
      // Need to copy as output is requested.
      *item->mutable_tracking_data() = *tracking_data;
    } else {
      item->mutable_tracking_data()->Swap(tracking_data.get());
    }

    const int next_chunk_msec =
        options_.caching_chunk_size_msec() * (chunk_idx_ + 1);

    if (timestamp.Value() / 1000 >= next_chunk_msec) {
      if (cc->Outputs().HasTag(kTrackingChunkTag)) {
        cc->Outputs()
            .Tag(kTrackingChunkTag)
            .Add(new TrackingDataChunk(tracking_chunk_),
                 Timestamp(tracking_chunk_.item(0).timestamp_usec()));
      }
      if (use_caching_) {
        WriteChunk(tracking_chunk_);
      }
      PrepareCurrentForNextChunk(&tracking_chunk_);
    }
  }

  if (cc->Outputs().HasTag(kTrackingTag)) {
    cc->Outputs()
        .Tag(kTrackingTag)
        .Add(tracking_data.release(), flow_stream->Value().Timestamp());
  }

  prev_timestamp_ = timestamp;
  ++frame_idx_;
  return absl::OkStatus();
}

absl::Status FlowPackagerCalculator::Close(CalculatorContext* cc) {
  if (frame_idx_ > 0) {
    tracking_chunk_.set_last_chunk(true);
    if (cc->Outputs().HasTag(kTrackingChunkTag)) {
      cc->Outputs()
          .Tag(kTrackingChunkTag)
          .Add(new TrackingDataChunk(tracking_chunk_),
               Timestamp(tracking_chunk_.item(0).timestamp_usec()));
    }

    if (use_caching_) {
      WriteChunk(tracking_chunk_);
    }
  }

  if (cc->Outputs().HasTag(kCompleteTag)) {
    cc->Outputs().Tag(kCompleteTag).Add(new bool(true), Timestamp::PreStream());
  }

  return absl::OkStatus();
}

void FlowPackagerCalculator::WriteChunk(const TrackingDataChunk& chunk) const {
  if (chunk.item_size() == 0) {
    LOG(ERROR) << "Write chunk called with empty tracking data."
               << "This can only occur if the spacing between frames "
               << "is larger than the requested chunk size. Try increasing "
               << "the chunk size";
    return;
  }

  auto format_runtime =
      absl::ParsedFormat<'d'>::New(options_.cache_file_format());

  std::string chunk_file;
  if (format_runtime) {
    chunk_file =
        cache_dir_ + "/" + absl::StrFormat(*format_runtime, chunk_idx_);
  } else {
    LOG(ERROR) << "chache_file_format wrong. fall back to chunk_%04d.";
    chunk_file = cache_dir_ + "/" + absl::StrFormat("chunk_%04d", chunk_idx_);
  }

  std::string data;
  chunk.SerializeToString(&data);

  const char* temp_filename = tempnam(cache_dir_.c_str(), nullptr);
  std::ofstream out_file(temp_filename);
  if (!out_file) {
    LOG(ERROR) << "Could not open " << temp_filename;
  } else {
    out_file.write(data.data(), data.size());
  }

  if (rename(temp_filename, chunk_file.c_str()) != 0) {
    LOG(ERROR) << "Failed to rename to " << chunk_file;
  }

  LOG(INFO) << "Wrote chunk : " << chunk_file;
}

void FlowPackagerCalculator::PrepareCurrentForNextChunk(
    TrackingDataChunk* chunk) {
  CHECK(chunk);
  if (chunk->item_size() == 0) {
    LOG(ERROR) << "Called with empty chunk. Unexpected.";
    return;
  }

  chunk->set_first_chunk(false);

  // Buffer last item for next chunk.
  TrackingDataChunk::Item last_item;
  last_item.Swap(chunk->mutable_item(chunk->item_size() - 1));

  chunk->Clear();
  chunk->add_item()->Swap(&last_item);

  ++chunk_idx_;
}

}  // namespace mediapipe
