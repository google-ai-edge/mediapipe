// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/modules/objectron/calculators/frame_annotation_tracker.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {
namespace {

// Create a new object annotation by shifting a reference
// object annotation.
ObjectAnnotation ShiftObject2d(const ObjectAnnotation& ref_obj, float dx,
                               float dy) {
  ObjectAnnotation obj = ref_obj;
  for (auto& keypoint : *(obj.mutable_keypoints())) {
    const float ref_x = keypoint.point_2d().x();
    const float ref_y = keypoint.point_2d().y();
    keypoint.mutable_point_2d()->set_x(ref_x + dx);
    keypoint.mutable_point_2d()->set_y(ref_y + dy);
  }
  return obj;
}

TimedBoxProto ShiftBox(const TimedBoxProto& ref_box, float dx, float dy) {
  TimedBoxProto box = ref_box;
  box.set_top(ref_box.top() + dy);
  box.set_bottom(ref_box.bottom() + dy);
  box.set_left(ref_box.left() + dx);
  box.set_right(ref_box.right() + dx);
  return box;
}

// Constructs a fixed ObjectAnnotation.
ObjectAnnotation ConstructFixedObject(
    const std::vector<std::vector<float>>& points) {
  ObjectAnnotation obj;
  for (const auto& point : points) {
    auto* keypoint = obj.add_keypoints();
    ABSL_CHECK_EQ(2, point.size());
    keypoint->mutable_point_2d()->set_x(point[0]);
    keypoint->mutable_point_2d()->set_y(point[1]);
  }
  return obj;
}

TEST(FrameAnnotationTrackerTest, TestConsolidation) {
  // Add 4 detections represented by FrameAnnotation, of which 3 correspond
  // to the same object.
  ObjectAnnotation object1, object2, object3, object4;
  // The bounding rectangle for these object keypoints is:
  // x: [0.2, 0.5], y: [0.1, 0.4]
  object3 = ConstructFixedObject({{0.35f, 0.25f},
                                  {0.3f, 0.3f},
                                  {0.2f, 0.4f},
                                  {0.3f, 0.1f},
                                  {0.2f, 0.2f},
                                  {0.5f, 0.3f},
                                  {0.4f, 0.4f},
                                  {0.5f, 0.1f},
                                  {0.4f, 0.2f}});
  object3.set_object_id(3);
  object1 = ShiftObject2d(object3, -0.05f, -0.05f);
  object1.set_object_id(1);
  object2 = ShiftObject2d(object3, 0.05f, 0.05f);
  object2.set_object_id(2);
  object4 = ShiftObject2d(object3, 0.2f, 0.2f);
  object4.set_object_id(4);
  FrameAnnotation frame_annotation_1;
  frame_annotation_1.set_timestamp(30 * 1000);  // 30ms
  *(frame_annotation_1.add_annotations()) = object1;
  *(frame_annotation_1.add_annotations()) = object4;
  FrameAnnotation frame_annotation_2;
  frame_annotation_2.set_timestamp(60 * 1000);  // 60ms
  *(frame_annotation_2.add_annotations()) = object2;
  FrameAnnotation frame_annotation_3;
  frame_annotation_3.set_timestamp(90 * 1000);  // 90ms
  *(frame_annotation_3.add_annotations()) = object3;

  FrameAnnotationTracker frame_annotation_tracker(/*iou_threshold*/ 0.5f, 1.0f,
                                                  1.0f);
  frame_annotation_tracker.AddDetectionResult(frame_annotation_1);
  frame_annotation_tracker.AddDetectionResult(frame_annotation_2);
  frame_annotation_tracker.AddDetectionResult(frame_annotation_3);

  TimedBoxProtoList timed_box_proto_list;
  TimedBoxProto* timed_box_proto = timed_box_proto_list.add_box();
  timed_box_proto->set_top(0.4f);
  timed_box_proto->set_bottom(0.7f);
  timed_box_proto->set_left(0.6f);
  timed_box_proto->set_right(0.9f);
  timed_box_proto->set_id(3);
  timed_box_proto->set_time_msec(150);
  timed_box_proto = timed_box_proto_list.add_box();
  *timed_box_proto = ShiftBox(timed_box_proto_list.box(0), 0.01f, 0.01f);
  timed_box_proto->set_id(1);
  timed_box_proto->set_time_msec(150);
  timed_box_proto = timed_box_proto_list.add_box();
  *timed_box_proto = ShiftBox(timed_box_proto_list.box(0), -0.01f, -0.01f);
  timed_box_proto->set_id(2);
  timed_box_proto->set_time_msec(150);
  absl::flat_hash_set<int> cancel_object_ids;
  FrameAnnotation tracked_detection =
      frame_annotation_tracker.ConsolidateTrackingResult(timed_box_proto_list,
                                                         &cancel_object_ids);
  EXPECT_EQ(2, cancel_object_ids.size());
  EXPECT_EQ(1, cancel_object_ids.count(1));
  EXPECT_EQ(1, cancel_object_ids.count(2));
  EXPECT_EQ(1, tracked_detection.annotations_size());
  EXPECT_EQ(3, tracked_detection.annotations(0).object_id());
  EXPECT_EQ(object3.keypoints_size(),
            tracked_detection.annotations(0).keypoints_size());
  const float x_offset = 0.4f;
  const float y_offset = 0.3f;
  const float tolerance = 1e-5f;
  for (int i = 0; i < object3.keypoints_size(); ++i) {
    const auto& point_2d =
        tracked_detection.annotations(0).keypoints(i).point_2d();
    EXPECT_NEAR(point_2d.x(), object3.keypoints(i).point_2d().x() + x_offset,
                tolerance);
    EXPECT_NEAR(point_2d.y(), object3.keypoints(i).point_2d().y() + y_offset,
                tolerance);
  }
}

}  // namespace
}  // namespace mediapipe
