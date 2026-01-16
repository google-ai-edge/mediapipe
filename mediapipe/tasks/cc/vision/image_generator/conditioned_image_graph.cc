/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/core/get_vector_item_calculator.h"
#include "mediapipe/calculators/core/get_vector_item_calculator.pb.h"
#include "mediapipe/calculators/util/flat_color_image_calculator.pb.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/conditioned_image_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/image_frame_util.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {

namespace internal {

// Helper postprocessing calculator for depth condition type to scale raw depth
// inference result to 0-255 uint8.
class DepthImagePostprocessingCalculator : public api2::Node {
 public:
  static constexpr api2::Input<Image> kImageIn{"IMAGE"};
  static constexpr api2::Output<Image> kImageOut{"IMAGE"};

  MEDIAPIPE_NODE_CONTRACT(kImageIn, kImageOut);

  absl::Status Process(CalculatorContext* cc) final {
    if (kImageIn(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    Image raw_depth_image = kImageIn(cc).Get();
    cv::Mat raw_depth_mat = mediapipe::formats::MatView(
        raw_depth_image.GetImageFrameSharedPtr().get());
    cv::Mat depth_mat;
    cv::normalize(raw_depth_mat, depth_mat, 255, 0, cv::NORM_MINMAX);
    depth_mat.convertTo(depth_mat, CV_8UC3, 1, 0);
    cv::cvtColor(depth_mat, depth_mat, cv::COLOR_GRAY2RGB);
    // Acquires the cv::Mat data and assign to the image frame.
    ImageFrameSharedPtr depth_image_frame_ptr = std::make_shared<ImageFrame>(
        mediapipe::ImageFormat::SRGB, depth_mat.cols, depth_mat.rows,
        depth_mat.step, depth_mat.data,
        [depth_mat](uint8_t[]) { depth_mat.~Mat(); });
    Image depth_image(depth_image_frame_ptr);
    kImageOut(cc).Send(depth_image);
    return absl::OkStatus();
  }
};

// NOLINTBEGIN: Node registration doesn't work when part of calculator name is
// moved to next line.
// clang-format off
MEDIAPIPE_REGISTER_NODE(::mediapipe::tasks::vision::image_generator::internal::DepthImagePostprocessingCalculator);
// clang-format on
// NOLINTEND

// Calculator to detect edges in the image with OpenCV Canny edge detection.
class CannyEdgeCalculator : public api2::Node {
 public:
  static constexpr api2::Input<Image> kImageIn{"IMAGE"};
  static constexpr api2::Output<Image> kImageOut{"IMAGE"};

  MEDIAPIPE_NODE_CONTRACT(kImageIn, kImageOut);

  absl::Status Process(CalculatorContext* cc) final {
    if (kImageIn(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    Image input_image = kImageIn(cc).Get();
    cv::Mat input_image_mat =
        mediapipe::formats::MatView(input_image.GetImageFrameSharedPtr().get());
    const auto& options = cc->Options<
        proto::ConditionedImageGraphOptions::EdgeConditionTypeOptions>();
    cv::Mat lumincance;
    cv::cvtColor(input_image_mat, lumincance, cv::COLOR_RGB2GRAY);
    cv::Mat edges_mat;
    cv::Canny(lumincance, edges_mat, options.threshold_1(),
              options.threshold_2(), options.aperture_size(),
              options.l2_gradient());
    cv::normalize(edges_mat, edges_mat, 255, 0, cv::NORM_MINMAX);
    edges_mat.convertTo(edges_mat, CV_8UC3, 1, 0);
    cv::cvtColor(edges_mat, edges_mat, cv::COLOR_GRAY2RGB);
    // Acquires the cv::Mat data and assign to the image frame.
    ImageFrameSharedPtr edges_image_frame_ptr = std::make_shared<ImageFrame>(
        mediapipe::ImageFormat::SRGB, edges_mat.cols, edges_mat.rows,
        edges_mat.step, edges_mat.data,
        [edges_mat](uint8_t[]) { edges_mat.~Mat(); });
    Image edges_image(edges_image_frame_ptr);
    kImageOut(cc).Send(edges_image);
    return absl::OkStatus();
  }
};

// NOLINTBEGIN: Node registration doesn't work when part of calculator name is
// moved to next line.
// clang-format off
MEDIAPIPE_REGISTER_NODE(::mediapipe::tasks::vision::image_generator::internal::CannyEdgeCalculator);
// clang-format on
// NOLINTEND

}  // namespace internal

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;

constexpr absl::string_view kImageTag = "IMAGE";
constexpr absl::string_view kUImageTag = "UIMAGE";
constexpr absl::string_view kNormLandmarksTag = "NORM_LANDMARKS";
constexpr absl::string_view kVectorTag = "VECTOR";
constexpr absl::string_view kItemTag = "ITEM";
constexpr absl::string_view kRenderDataTag = "RENDER_DATA";
constexpr absl::string_view kConfidenceMaskTag = "CONFIDENCE_MASK:0";

enum ColorType {
  WHITE = 0,
  GREEN = 1,
  RED = 2,
  BLACK = 3,
  BLUE = 4,
};

mediapipe::Color GetColor(ColorType color_type) {
  mediapipe::Color color;
  switch (color_type) {
    case WHITE:
      color.set_b(255);
      color.set_g(255);
      color.set_r(255);
      break;
    case GREEN:
      color.set_b(0);
      color.set_g(255);
      color.set_r(0);
      break;
    case RED:
      color.set_b(0);
      color.set_g(0);
      color.set_r(255);
      break;
    case BLACK:
      color.set_b(0);
      color.set_g(0);
      color.set_r(0);
      break;
    case BLUE:
      color.set_b(255);
      color.set_g(0);
      color.set_r(0);
      break;
  }
  return color;
}

// Get LandmarksToRenderDataCalculatorOptions for rendering face landmarks
// connections.
mediapipe::LandmarksToRenderDataCalculatorOptions
GetFaceLandmarksRenderDataOptions(
    absl::Span<const std::array<int, 2>> connections, ColorType color_type) {
  mediapipe::LandmarksToRenderDataCalculatorOptions render_options;
  render_options.set_thickness(1);
  render_options.set_visualize_landmark_depth(false);
  render_options.set_render_landmarks(false);
  *render_options.mutable_connection_color() = GetColor(color_type);
  for (const auto& connection : connections) {
    render_options.add_landmark_connections(connection[0]);
    render_options.add_landmark_connections(connection[1]);
  }
  return render_options;
}

Source<mediapipe::RenderData> GetFaceLandmarksRenderData(
    Source<mediapipe::NormalizedLandmarkList> face_landmarks,
    const mediapipe::LandmarksToRenderDataCalculatorOptions&
        landmarks_to_render_data_options,
    Graph& graph) {
  auto& landmarks_to_render_data =
      graph.AddNode("LandmarksToRenderDataCalculator");
  landmarks_to_render_data
      .GetOptions<mediapipe::LandmarksToRenderDataCalculatorOptions>()
      .CopyFrom(landmarks_to_render_data_options);
  face_landmarks >> landmarks_to_render_data.In(kNormLandmarksTag);
  return landmarks_to_render_data.Out(kRenderDataTag)
      .Cast<mediapipe::RenderData>();
}

// Add FaceLandmarkerGraph to detect the face landmarks in the given face image,
// and generate a face mesh guidance image for the diffusion plugin model.
absl::StatusOr<Source<Image>> GetFaceLandmarksImage(
    Source<Image> face_image,
    const proto::ConditionedImageGraphOptions::FaceConditionTypeOptions&
        face_condition_type_options,
    Graph& graph) {
  if (face_condition_type_options.face_landmarker_graph_options()
          .face_detector_graph_options()
          .num_faces() != 1) {
    return absl::InvalidArgumentError(
        "Only supports face landmarks of a single face as the guidance image.");
  }

  // Detect face landmarks.
  auto& face_landmarker_graph = graph.AddNode(
      "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph");
  face_landmarker_graph
      .GetOptions<face_landmarker::proto::FaceLandmarkerGraphOptions>()
      .CopyFrom(face_condition_type_options.face_landmarker_graph_options());
  face_image >> face_landmarker_graph.In(kImageTag);
  auto face_landmarks_lists =
      face_landmarker_graph.Out(kNormLandmarksTag)
          .Cast<std::vector<mediapipe::NormalizedLandmarkList>>();

  // Get the single face landmarks.
  auto& get_vector_item =
      graph.AddNode("GetNormalizedLandmarkListVectorItemCalculator");
  get_vector_item.GetOptions<mediapipe::GetVectorItemCalculatorOptions>()
      .set_item_index(0);
  face_landmarks_lists >> get_vector_item.In(kVectorTag);
  auto single_face_landmarks =
      get_vector_item.Out(kItemTag).Cast<mediapipe::NormalizedLandmarkList>();

  // Convert face landmarks to render data.
  auto face_oval = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksFaceOval
                  .data(),
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksFaceOval
                  .size()),
          ColorType::WHITE),
      graph);
  auto lips = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksLips
                  .data(),
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksLips
                  .size()),
          ColorType::WHITE),
      graph);
  auto left_eye = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksLeftEye
                  .data(),
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksLeftEye
                  .size()),
          ColorType::GREEN),
      graph);
  auto left_eye_brow = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::
                  kFaceLandmarksLeftEyeBrow.data(),
              face_landmarker::FaceLandmarksConnections::
                  kFaceLandmarksLeftEyeBrow.size()),
          ColorType::GREEN),
      graph);
  auto left_iris = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksLeftIris
                  .data(),
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksLeftIris
                  .size()),
          ColorType::GREEN),
      graph);

  auto right_eye = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksRightEye
                  .data(),
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksRightEye
                  .size()),
          ColorType::BLUE),
      graph);
  auto right_eye_brow = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::
                  kFaceLandmarksRightEyeBrow.data(),
              face_landmarker::FaceLandmarksConnections::
                  kFaceLandmarksRightEyeBrow.size()),
          ColorType::BLUE),
      graph);
  auto right_iris = GetFaceLandmarksRenderData(
      single_face_landmarks,
      GetFaceLandmarksRenderDataOptions(
          absl::Span<const std::array<int, 2>>(
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksRightIris
                  .data(),
              face_landmarker::FaceLandmarksConnections::kFaceLandmarksRightIris
                  .size()),
          ColorType::BLUE),
      graph);

  // Create a black canvas image with same size as face image.
  auto& flat_color = graph.AddNode("FlatColorImageCalculator");
  flat_color.GetOptions<mediapipe::FlatColorImageCalculatorOptions>()
      .mutable_color()
      ->set_r(0);
  face_image >> flat_color.In(kImageTag);
  auto blank_canvas = flat_color.Out(kImageTag);

  // Draw render data on the canvas image.
  auto& annotation_overlay = graph.AddNode("AnnotationOverlayCalculator");
  blank_canvas >> annotation_overlay.In(kUImageTag);
  face_oval >> annotation_overlay.In(0);
  lips >> annotation_overlay.In(1);
  left_eye >> annotation_overlay.In(2);
  left_eye_brow >> annotation_overlay.In(3);
  left_iris >> annotation_overlay.In(4);
  right_eye >> annotation_overlay.In(5);
  right_eye_brow >> annotation_overlay.In(6);
  right_iris >> annotation_overlay.In(7);
  return annotation_overlay.Out(kUImageTag).Cast<Image>();
}

absl::StatusOr<Source<Image>> GetDepthImage(
    Source<Image> image,
    const image_generator::proto::ConditionedImageGraphOptions::
        DepthConditionTypeOptions& depth_condition_type_options,
    Graph& graph) {
  auto& image_segmenter_graph = graph.AddNode(
      "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph");
  image_segmenter_graph
      .GetOptions<image_segmenter::proto::ImageSegmenterGraphOptions>()
      .CopyFrom(depth_condition_type_options.image_segmenter_graph_options());
  image >> image_segmenter_graph.In(kImageTag);
  auto raw_depth_image = image_segmenter_graph.Out(kConfidenceMaskTag);

  auto& depth_postprocessing = graph.AddNode(
      "mediapipe.tasks.vision.image_generator.internal."
      "DepthImagePostprocessingCalculator");
  raw_depth_image >> depth_postprocessing.In(kImageTag);
  return depth_postprocessing.Out(kImageTag).Cast<Image>();
}

absl::StatusOr<Source<Image>> GetEdgeImage(
    Source<Image> image,
    const image_generator::proto::ConditionedImageGraphOptions::
        EdgeConditionTypeOptions& edge_condition_type_options,
    Graph& graph) {
  auto& edge_detector = graph.AddNode(
      "mediapipe.tasks.vision.image_generator.internal."
      "CannyEdgeCalculator");
  edge_detector
      .GetOptions<
          proto::ConditionedImageGraphOptions::EdgeConditionTypeOptions>()
      .CopyFrom(edge_condition_type_options);
  image >> edge_detector.In(kImageTag);
  return edge_detector.Out(kImageTag).Cast<Image>();
}

}  // namespace

// A mediapipe.tasks.vision.image_generator.ConditionedImageGraph converts the
// input image to an image of condition type. The output image can be used as
// input for the diffusion model with control plugin.
// Inputs:
//   IMAGE - Image
//     Conditioned image to generate the image for diffusion plugin model.
//
// Outputs:
//   IMAGE - Image
//     The guidance image used as input for the diffusion plugin model.
class ConditionedImageGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    auto& graph_options =
        *sc->MutableOptions<proto::ConditionedImageGraphOptions>();
    Source<Image> conditioned_image = graph.In(kImageTag).Cast<Image>();
    // Configure the guidance graph and get the guidance image if guidance graph
    // options is set.
    switch (graph_options.condition_type_options_case()) {
      case proto::ConditionedImageGraphOptions::CONDITION_TYPE_OPTIONS_NOT_SET:
        return absl::InvalidArgumentError(
            "Conditioned type options is not set.");
        break;
      case proto::ConditionedImageGraphOptions::kFaceConditionTypeOptions: {
        MP_ASSIGN_OR_RETURN(
            auto face_landmarks_image,
            GetFaceLandmarksImage(conditioned_image,
                                  graph_options.face_condition_type_options(),
                                  graph));
        face_landmarks_image >> graph.Out(kImageTag);
      } break;
      case proto::ConditionedImageGraphOptions::kDepthConditionTypeOptions: {
        MP_ASSIGN_OR_RETURN(
            auto depth_image,
            GetDepthImage(conditioned_image,
                          graph_options.depth_condition_type_options(), graph));
        depth_image >> graph.Out(kImageTag);
      } break;
      case proto::ConditionedImageGraphOptions::kEdgeConditionTypeOptions: {
        MP_ASSIGN_OR_RETURN(
            auto edges_image,
            GetEdgeImage(conditioned_image,
                         graph_options.edge_condition_type_options(), graph));
        edges_image >> graph.Out(kImageTag);
      } break;
    }
    return graph.GetConfig();
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::image_generator::ConditionedImageGraph);

}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
