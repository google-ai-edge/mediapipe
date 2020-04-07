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
#include <unordered_set>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "mediapipe/calculators/video/box_detector_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_features2d_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/util/tracking/box_detector.h"
#include "mediapipe/util/tracking/box_tracker.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"
#include "mediapipe/util/tracking/tracking.h"
#include "mediapipe/util/tracking/tracking_visualization_utilities.h"

#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

// A calculator to detect reappeared box positions from single frame.
//
// Input stream:
//   TRACKING: Input tracking data (proto TrackingData) containing features and
//             descriptors.
//   VIDEO:    Optional input video stream tracked boxes are rendered over
//             (Required if VIZ is specified).
//   FEATURES: Input feature points (std::vector<cv::KeyPoint>) in the original
//             pixel space.
//   DESCRIPTORS: Input feature descriptors (std::vector<float>). Actual feature
//             dimension needs to be specified in detector_options.
//   IMAGE_SIZE: Input image dimension.
//   TRACKED_BOXES : input box tracking result (proto TimedBoxProtoList) from
//             BoxTrackerCalculator.
//   ADD_INDEX: Optional std::string containing binary format proto of type
//             BoxDetectorIndex. Used for adding target index to the detector
//             search index during runtime.
//   CANCEL_OBJECT_ID: Optional id of box to be removed. This is recommended
//             to be used with SyncSetInputStreamHandler.
//   REACQ_SWITCH: Optional bool for swithcing on and off reacquisition
//             functionality. User should initialize a graph with box detector
//             calculator and be able to switch it on and off in runtime.
//
// Output streams:
//   VIZ:   Optional output video stream with rendered box positions
//          (requires VIDEO to be present)
//   BOXES: Optional output stream of type TimedBoxProtoList for each lost box.
//
// Imput side packets:
//   INDEX_PROTO_STRING: Optional std::string containing binary format proto of
//   type
//                       BoxDetectorIndex. Used for initializing box_detector
//                       with predefined template images.
//   FRAME_ALIGNMENT:    Optional integer to indicate alignment_boundary for
//                       outputing ImageFrame in "VIZ" stream.
//                       Set to ImageFrame::kDefaultAlignmentBoundary for
//                       offline pipeline to be compatible with FFmpeg.
//                       Set to ImageFrame::kGlDefaultAlignmentBoundary for Apps
//                       to be compatible with GL renderer.
//   OUTPUT_INDEX_FILENAME: File path to the output index file.

class BoxDetectorCalculator : public CalculatorBase {
 public:
  ~BoxDetectorCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  BoxDetectorCalculatorOptions options_;
  std::unique_ptr<BoxDetectorInterface> box_detector_;
  bool detector_switch_ = true;
  uint32 frame_alignment_ = ImageFrame::kDefaultAlignmentBoundary;
  bool write_index_ = false;
  int box_id_ = 0;
};

REGISTER_CALCULATOR(BoxDetectorCalculator);

::mediapipe::Status BoxDetectorCalculator::GetContract(CalculatorContract* cc) {
  if (cc->Inputs().HasTag("TRACKING")) {
    cc->Inputs().Tag("TRACKING").Set<TrackingData>();
  }

  if (cc->Inputs().HasTag("TRACKED_BOXES")) {
    cc->Inputs().Tag("TRACKED_BOXES").Set<TimedBoxProtoList>();
  }

  if (cc->Inputs().HasTag("VIDEO")) {
    cc->Inputs().Tag("VIDEO").Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag("FEATURES")) {
    RET_CHECK(cc->Inputs().HasTag("DESCRIPTORS"))
        << "FEATURES and DESCRIPTORS need to be specified together.";
    cc->Inputs().Tag("FEATURES").Set<std::vector<cv::KeyPoint>>();
  }

  if (cc->Inputs().HasTag("DESCRIPTORS")) {
    RET_CHECK(cc->Inputs().HasTag("FEATURES"))
        << "FEATURES and DESCRIPTORS need to be specified together.";
    cc->Inputs().Tag("DESCRIPTORS").Set<std::vector<float>>();
  }

  if (cc->Inputs().HasTag("IMAGE_SIZE")) {
    cc->Inputs().Tag("IMAGE_SIZE").Set<std::pair<int, int>>();
  }

  if (cc->Inputs().HasTag("ADD_INDEX")) {
    cc->Inputs().Tag("ADD_INDEX").Set<std::string>();
  }

  if (cc->Inputs().HasTag("CANCEL_OBJECT_ID")) {
    cc->Inputs().Tag("CANCEL_OBJECT_ID").Set<int>();
  }

  if (cc->Inputs().HasTag("REACQ_SWITCH")) {
    cc->Inputs().Tag("REACQ_SWITCH").Set<bool>();
  }

  if (cc->Outputs().HasTag("BOXES")) {
    cc->Outputs().Tag("BOXES").Set<TimedBoxProtoList>();
  }

  if (cc->Outputs().HasTag("VIZ")) {
    RET_CHECK(cc->Inputs().HasTag("VIDEO"))
        << "Output stream VIZ requires VIDEO to be present.";
    cc->Outputs().Tag("VIZ").Set<ImageFrame>();
  }

  if (cc->InputSidePackets().HasTag("INDEX_PROTO_STRING")) {
    cc->InputSidePackets().Tag("INDEX_PROTO_STRING").Set<std::string>();
  }

  if (cc->InputSidePackets().HasTag("OUTPUT_INDEX_FILENAME")) {
    cc->InputSidePackets().Tag("OUTPUT_INDEX_FILENAME").Set<std::string>();
  }

  if (cc->InputSidePackets().HasTag("FRAME_ALIGNMENT")) {
    cc->InputSidePackets().Tag("FRAME_ALIGNMENT").Set<int>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BoxDetectorCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<BoxDetectorCalculatorOptions>();
  box_detector_ = BoxDetectorInterface::Create(options_.detector_options());

  if (cc->InputSidePackets().HasTag("INDEX_PROTO_STRING")) {
    BoxDetectorIndex predefined_index;
    if (!predefined_index.ParseFromString(cc->InputSidePackets()
                                              .Tag("INDEX_PROTO_STRING")
                                              .Get<std::string>())) {
      LOG(FATAL) << "failed to parse BoxDetectorIndex from INDEX_PROTO_STRING";
    }
    box_detector_->AddBoxDetectorIndex(predefined_index);
  }

  for (const auto& filename : options_.index_proto_filename()) {
    std::string string_path;
    ASSIGN_OR_RETURN(string_path, PathToResourceAsFile(filename));
    std::string index_string;
    MP_RETURN_IF_ERROR(file::GetContents(string_path, &index_string));
    BoxDetectorIndex predefined_index;
    if (!predefined_index.ParseFromString(index_string)) {
      LOG(FATAL)
          << "failed to parse BoxDetectorIndex from index_proto_filename";
    }
    box_detector_->AddBoxDetectorIndex(predefined_index);
  }

  if (cc->InputSidePackets().HasTag("OUTPUT_INDEX_FILENAME")) {
    write_index_ = true;
  }

  if (cc->InputSidePackets().HasTag("FRAME_ALIGNMENT")) {
    frame_alignment_ = cc->InputSidePackets().Tag("FRAME_ALIGNMENT").Get<int>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BoxDetectorCalculator::Process(CalculatorContext* cc) {
  const Timestamp timestamp = cc->InputTimestamp();
  const int64 timestamp_msec = timestamp.Value() / 1000;

  InputStream* cancel_object_id_stream =
      cc->Inputs().HasTag("CANCEL_OBJECT_ID")
          ? &(cc->Inputs().Tag("CANCEL_OBJECT_ID"))
          : nullptr;
  if (cancel_object_id_stream && !cancel_object_id_stream->IsEmpty()) {
    const int cancel_object_id = cancel_object_id_stream->Get<int>();
    box_detector_->CancelBoxDetection(cancel_object_id);
  }

  InputStream* add_index_stream = cc->Inputs().HasTag("ADD_INDEX")
                                      ? &(cc->Inputs().Tag("ADD_INDEX"))
                                      : nullptr;
  if (add_index_stream && !add_index_stream->IsEmpty()) {
    BoxDetectorIndex predefined_index;
    if (!predefined_index.ParseFromString(
            add_index_stream->Get<std::string>())) {
      LOG(FATAL) << "failed to parse BoxDetectorIndex from ADD_INDEX";
    }
    box_detector_->AddBoxDetectorIndex(predefined_index);
  }

  InputStream* reacq_switch_stream = cc->Inputs().HasTag("REACQ_SWITCH")
                                         ? &(cc->Inputs().Tag("REACQ_SWITCH"))
                                         : nullptr;
  if (reacq_switch_stream && !reacq_switch_stream->IsEmpty()) {
    detector_switch_ = reacq_switch_stream->Get<bool>();
  }

  if (!detector_switch_) {
    return ::mediapipe::OkStatus();
  }

  InputStream* track_stream = cc->Inputs().HasTag("TRACKING")
                                  ? &(cc->Inputs().Tag("TRACKING"))
                                  : nullptr;
  InputStream* video_stream =
      cc->Inputs().HasTag("VIDEO") ? &(cc->Inputs().Tag("VIDEO")) : nullptr;
  InputStream* feature_stream = cc->Inputs().HasTag("FEATURES")
                                    ? &(cc->Inputs().Tag("FEATURES"))
                                    : nullptr;
  InputStream* descriptor_stream = cc->Inputs().HasTag("DESCRIPTORS")
                                       ? &(cc->Inputs().Tag("DESCRIPTORS"))
                                       : nullptr;

  CHECK(track_stream != nullptr || video_stream != nullptr ||
        (feature_stream != nullptr && descriptor_stream != nullptr))
      << "One and only one of {tracking_data, input image frame, "
         "feature/descriptor} need to be valid.";

  InputStream* tracked_boxes_stream = cc->Inputs().HasTag("TRACKED_BOXES")
                                          ? &(cc->Inputs().Tag("TRACKED_BOXES"))
                                          : nullptr;
  std::unique_ptr<TimedBoxProtoList> detected_boxes(new TimedBoxProtoList());

  if (track_stream != nullptr) {
    // Detect from tracking data
    if (track_stream->IsEmpty()) {
      return ::mediapipe::OkStatus();
    }

    const TrackingData& tracking_data = track_stream->Get<TrackingData>();

    CHECK(tracked_boxes_stream != nullptr) << "tracked_boxes needed.";

    const TimedBoxProtoList tracked_boxes =
        tracked_boxes_stream->Get<TimedBoxProtoList>();

    box_detector_->DetectAndAddBox(tracking_data, tracked_boxes, timestamp_msec,
                                   detected_boxes.get());
  } else if (video_stream != nullptr) {
    // Detect from input frame
    if (video_stream->IsEmpty()) {
      return ::mediapipe::OkStatus();
    }

    TimedBoxProtoList tracked_boxes;
    if (tracked_boxes_stream != nullptr && !tracked_boxes_stream->IsEmpty()) {
      tracked_boxes = tracked_boxes_stream->Get<TimedBoxProtoList>();
    }

    // Just directly pass along the image frame data as-is for detection; we
    // don't need to worry about conforming to a specific alignment here.
    const cv::Mat input_view =
        formats::MatView(&video_stream->Get<ImageFrame>());
    box_detector_->DetectAndAddBox(input_view, tracked_boxes, timestamp_msec,
                                   detected_boxes.get());
  } else {
    if (feature_stream->IsEmpty() || descriptor_stream->IsEmpty()) {
      return ::mediapipe::OkStatus();
    }

    const auto& image_size =
        cc->Inputs().Tag("IMAGE_SIZE").Get<std::pair<int, int>>();
    float inv_scale = 1.0f / std::max(image_size.first, image_size.second);

    TimedBoxProtoList tracked_boxes;
    if (tracked_boxes_stream != nullptr && !tracked_boxes_stream->IsEmpty()) {
      tracked_boxes = tracked_boxes_stream->Get<TimedBoxProtoList>();
    } else if (write_index_) {
      auto* box_ptr = tracked_boxes.add_box();
      box_ptr->set_id(box_id_);
      box_ptr->set_reacquisition(true);
      box_ptr->set_aspect_ratio((float)image_size.first /
                                (float)image_size.second);

      box_ptr->mutable_quad()->add_vertices(0);
      box_ptr->mutable_quad()->add_vertices(0);

      box_ptr->mutable_quad()->add_vertices(0);
      box_ptr->mutable_quad()->add_vertices(1);

      box_ptr->mutable_quad()->add_vertices(1);
      box_ptr->mutable_quad()->add_vertices(1);

      box_ptr->mutable_quad()->add_vertices(1);
      box_ptr->mutable_quad()->add_vertices(0);

      ++box_id_;
    }

    const auto& features = feature_stream->Get<std::vector<cv::KeyPoint>>();
    const int feature_size = features.size();
    std::vector<Vector2_f> features_vec(feature_size);

    const auto& descriptors = descriptor_stream->Get<std::vector<float>>();
    const int dims = options_.detector_options().descriptor_dims();
    CHECK_GE(descriptors.size(), feature_size * dims);
    cv::Mat descriptors_mat(feature_size, dims, CV_32F);
    for (int j = 0; j < feature_size; ++j) {
      features_vec[j].Set(features[j].pt.x * inv_scale,
                          features[j].pt.y * inv_scale);
      for (int i = 0; i < dims; ++i) {
        descriptors_mat.at<float>(j, i) = descriptors[j * dims + i];
      }
    }

    box_detector_->DetectAndAddBoxFromFeatures(
        features_vec, descriptors_mat, tracked_boxes, timestamp_msec,
        image_size.first * inv_scale, image_size.second * inv_scale,
        detected_boxes.get());
  }

  if (cc->Outputs().HasTag("VIZ")) {
    cv::Mat viz_view;
    std::unique_ptr<ImageFrame> viz_frame;
    if (video_stream != nullptr && !video_stream->IsEmpty()) {
      viz_frame = absl::make_unique<ImageFrame>();
      viz_frame->CopyFrom(video_stream->Get<ImageFrame>(), frame_alignment_);
      viz_view = formats::MatView(viz_frame.get());
    }
    for (const auto& box : detected_boxes->box()) {
      RenderBox(box, &viz_view);
    }
    cc->Outputs().Tag("VIZ").Add(viz_frame.release(), timestamp);
  }

  if (cc->Outputs().HasTag("BOXES")) {
    cc->Outputs().Tag("BOXES").Add(detected_boxes.release(), timestamp);
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BoxDetectorCalculator::Close(CalculatorContext* cc) {
  if (write_index_) {
    BoxDetectorIndex index = box_detector_->ObtainBoxDetectorIndex();
    MEDIAPIPE_CHECK_OK(mediapipe::file::SetContents(
        cc->InputSidePackets().Tag("OUTPUT_INDEX_FILENAME").Get<std::string>(),
        index.SerializeAsString()));
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
