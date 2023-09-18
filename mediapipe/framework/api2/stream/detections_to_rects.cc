#include "mediapipe/framework/api2/stream/detections_to_rects.h"

#include <utility>
#include <vector>

#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe::api2::builder {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::builder::Graph;

void AddOptions(int start_keypoint_index, int end_keypoint_index,
                float target_angle,
                mediapipe::api2::builder::GenericNode& node) {
  auto& options = node.GetOptions<DetectionsToRectsCalculatorOptions>();
  options.set_rotation_vector_start_keypoint_index(start_keypoint_index);
  options.set_rotation_vector_end_keypoint_index(end_keypoint_index);
  options.set_rotation_vector_target_angle_degrees(target_angle);
}

}  // namespace

Stream<NormalizedRect> ConvertAlignmentPointsDetectionToRect(
    Stream<Detection> detection, Stream<std::pair<int, int>> image_size,
    int start_keypoint_index, int end_keypoint_index, float target_angle,
    Graph& graph) {
  auto& align_node = graph.AddNode("AlignmentPointsRectsCalculator");
  AddOptions(start_keypoint_index, end_keypoint_index, target_angle,
             align_node);
  detection.ConnectTo(align_node.In("DETECTION"));
  image_size.ConnectTo(align_node.In("IMAGE_SIZE"));
  return align_node.Out("NORM_RECT").Cast<NormalizedRect>();
}

Stream<NormalizedRect> ConvertAlignmentPointsDetectionsToRect(
    Stream<std::vector<Detection>> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle, Graph& graph) {
  auto& align_node = graph.AddNode("AlignmentPointsRectsCalculator");
  AddOptions(start_keypoint_index, end_keypoint_index, target_angle,
             align_node);
  detections.ConnectTo(align_node.In("DETECTIONS"));
  image_size.ConnectTo(align_node.In("IMAGE_SIZE"));
  return align_node.Out("NORM_RECT").Cast<NormalizedRect>();
}

Stream<NormalizedRect> ConvertDetectionToRect(
    Stream<Detection> detection, Stream<std::pair<int, int>> image_size,
    int start_keypoint_index, int end_keypoint_index, float target_angle,
    mediapipe::api2::builder::Graph& graph) {
  auto& align_node = graph.AddNode("DetectionsToRectsCalculator");
  AddOptions(start_keypoint_index, end_keypoint_index, target_angle,
             align_node);
  detection.ConnectTo(align_node.In("DETECTION"));
  image_size.ConnectTo(align_node.In("IMAGE_SIZE"));
  return align_node.Out("NORM_RECT").Cast<NormalizedRect>();
}

Stream<std::vector<NormalizedRect>> ConvertDetectionsToRects(
    Stream<std::vector<Detection>> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    mediapipe::api2::builder::Graph& graph) {
  // TODO: check if we can substitute DetectionsToRectsCalculator
  // with AlignmentPointsRectsCalculator and use it instead. Ideally, merge or
  // remove one of calculators.
  auto& align_node = graph.AddNode("DetectionsToRectsCalculator");
  AddOptions(start_keypoint_index, end_keypoint_index, target_angle,
             align_node);
  detections.ConnectTo(align_node.In("DETECTIONS"));
  image_size.ConnectTo(align_node.In("IMAGE_SIZE"));
  return align_node.Out("NORM_RECTS").Cast<std::vector<NormalizedRect>>();
}

Stream<NormalizedRect> ConvertDetectionsToRectUsingKeypoints(
    Stream<std::vector<Detection>> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    mediapipe::api2::builder::Graph& graph) {
  auto& node = graph.AddNode("DetectionsToRectsCalculator");

  auto& options = node.GetOptions<DetectionsToRectsCalculatorOptions>();
  options.set_rotation_vector_start_keypoint_index(start_keypoint_index);
  options.set_rotation_vector_end_keypoint_index(end_keypoint_index);
  options.set_rotation_vector_target_angle_degrees(target_angle);
  options.set_conversion_mode(
      DetectionsToRectsCalculatorOptions::USE_KEYPOINTS);

  detections.ConnectTo(node.In("DETECTIONS"));
  image_size.ConnectTo(node.In("IMAGE_SIZE"));
  return node.Out("NORM_RECT").Cast<NormalizedRect>();
}

}  // namespace mediapipe::api2::builder
