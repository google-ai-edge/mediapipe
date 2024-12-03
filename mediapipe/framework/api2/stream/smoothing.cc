#include "mediapipe/framework/api2/stream/smoothing.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/calculators/util/visibility_smoothing_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe::api2::builder {

namespace {

void SetFilterConfig(const OneEuroFilterConfig& config,
                     bool disable_value_scaling, GenericNode& node) {
  auto& smoothing_node_opts =
      node.GetOptions<LandmarksSmoothingCalculatorOptions>();
  auto& one_euro_filter = *smoothing_node_opts.mutable_one_euro_filter();
  one_euro_filter.set_min_cutoff(config.min_cutoff);
  one_euro_filter.set_derivate_cutoff(config.derivate_cutoff);
  one_euro_filter.set_beta(config.beta);
  one_euro_filter.set_disable_value_scaling(disable_value_scaling);
}

void SetFilterConfig(const LandmarksSmoothingCalculatorOptions& config,
                     GenericNode& node) {
  auto& smoothing_node_opts =
      node.GetOptions<LandmarksSmoothingCalculatorOptions>();
  smoothing_node_opts = config;
}

GenericNode& AddVisibilitySmoothingNode(float low_pass_filter_alpha,
                                        Graph& graph) {
  auto& smoothing_node = graph.AddNode("VisibilitySmoothingCalculator");
  auto& smoothing_node_opts =
      smoothing_node.GetOptions<VisibilitySmoothingCalculatorOptions>();
  smoothing_node_opts.mutable_low_pass_filter()->set_alpha(
      low_pass_filter_alpha);
  return smoothing_node;
}

}  // namespace

Stream<NormalizedLandmarkList> SmoothLandmarks(
    Stream<NormalizedLandmarkList> landmarks,
    Stream<std::pair<int, int>> image_size,
    std::optional<Stream<NormalizedRect>> scale_roi,
    const OneEuroFilterConfig& config, Graph& graph) {
  auto& smoothing_node = graph.AddNode("LandmarksSmoothingCalculator");
  SetFilterConfig(config, /*disable_value_scaling=*/false, smoothing_node);

  landmarks.ConnectTo(smoothing_node.In("NORM_LANDMARKS"));
  image_size.ConnectTo(smoothing_node.In("IMAGE_SIZE"));
  if (scale_roi) {
    scale_roi->ConnectTo(smoothing_node.In("OBJECT_SCALE_ROI"));
  }
  return smoothing_node.Out("NORM_FILTERED_LANDMARKS")
      .Cast<NormalizedLandmarkList>();
}

Stream<LandmarkList> SmoothLandmarks(
    Stream<LandmarkList> landmarks,
    std::optional<Stream<NormalizedRect>> scale_roi,
    const OneEuroFilterConfig& config, Graph& graph) {
  auto& smoothing_node = graph.AddNode("LandmarksSmoothingCalculator");
  SetFilterConfig(config, /*disable_value_scaling=*/true, smoothing_node);

  landmarks.ConnectTo(smoothing_node.In("LANDMARKS"));
  if (scale_roi) {
    scale_roi->ConnectTo(smoothing_node.In("OBJECT_SCALE_ROI"));
  }
  return smoothing_node.Out("FILTERED_LANDMARKS").Cast<LandmarkList>();
}

Stream<std::vector<NormalizedLandmarkList>> SmoothMultiLandmarks(
    Stream<std::vector<NormalizedLandmarkList>> landmarks,
    Stream<std::vector<int64_t>> tracking_ids,
    Stream<std::pair<int, int>> image_size,
    std::optional<Stream<std::vector<NormalizedRect>>> scale_roi,
    const LandmarksSmoothingCalculatorOptions& config, Graph& graph) {
  auto& smoothing_node = graph.AddNode("MultiLandmarksSmoothingCalculator");
  SetFilterConfig(config, smoothing_node);

  landmarks.ConnectTo(smoothing_node.In("NORM_LANDMARKS"));
  tracking_ids.ConnectTo(smoothing_node.In("TRACKING_IDS"));
  image_size.ConnectTo(smoothing_node.In("IMAGE_SIZE"));
  if (scale_roi) {
    scale_roi->ConnectTo(smoothing_node.In("OBJECT_SCALE_ROI"));
  }
  return smoothing_node.Out("NORM_FILTERED_LANDMARKS")
      .Cast<std::vector<NormalizedLandmarkList>>();
}

Stream<std::vector<LandmarkList>> SmoothMultiWorldLandmarks(
    Stream<std::vector<LandmarkList>> landmarks,
    Stream<std::vector<int64_t>> tracking_ids,
    std::optional<Stream<std::vector<Rect>>> scale_roi,
    const LandmarksSmoothingCalculatorOptions& config, Graph& graph) {
  auto& smoothing_node =
      graph.AddNode("MultiWorldLandmarksSmoothingCalculator");
  SetFilterConfig(config, smoothing_node);

  landmarks.ConnectTo(smoothing_node.In("LANDMARKS"));
  tracking_ids.ConnectTo(smoothing_node.In("TRACKING_IDS"));
  if (scale_roi) {
    scale_roi->ConnectTo(smoothing_node.In("OBJECT_SCALE_ROI"));
  }
  return smoothing_node.Out("FILTERED_LANDMARKS")
      .Cast<std::vector<LandmarkList>>();
}

Stream<NormalizedLandmarkList> SmoothLandmarksVisibility(
    Stream<NormalizedLandmarkList> landmarks, float low_pass_filter_alpha,
    Graph& graph) {
  auto& node = AddVisibilitySmoothingNode(low_pass_filter_alpha, graph);
  landmarks.ConnectTo(node.In("NORM_LANDMARKS"));
  return node.Out("NORM_FILTERED_LANDMARKS").Cast<NormalizedLandmarkList>();
}

Stream<LandmarkList> SmoothLandmarksVisibility(Stream<LandmarkList> landmarks,
                                               float low_pass_filter_alpha,
                                               Graph& graph) {
  auto& node = AddVisibilitySmoothingNode(low_pass_filter_alpha, graph);
  landmarks.ConnectTo(node.In("LANDMARKS"));
  return node.Out("FILTERED_LANDMARKS").Cast<LandmarkList>();
}

}  // namespace mediapipe::api2::builder
