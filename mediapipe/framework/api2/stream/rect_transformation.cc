#include "mediapipe/framework/api2/stream/rect_transformation.h"

#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe::api2::builder {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;

template <typename TransformeeT>
Stream<TransformeeT> InternalScaleAndShift(
    Stream<TransformeeT> transformee, Stream<std::pair<int, int>> image_size,
    float scale_x_factor, float scale_y_factor, std::optional<float> shift_x,
    std::optional<float> shift_y, bool square_long, Graph& graph) {
  auto& node = graph.AddNode("RectTransformationCalculator");
  auto& node_opts =
      node.GetOptions<mediapipe::RectTransformationCalculatorOptions>();
  node_opts.set_scale_x(scale_x_factor);
  node_opts.set_scale_y(scale_y_factor);
  if (shift_x) {
    node_opts.set_shift_x(shift_x.value());
  }
  if (shift_y) {
    node_opts.set_shift_y(shift_y.value());
  }
  if (square_long) {
    node_opts.set_square_long(square_long);
  }
  image_size.ConnectTo(node.In("IMAGE_SIZE"));
  if constexpr (std::is_same_v<TransformeeT, std::vector<NormalizedRect>>) {
    transformee.ConnectTo(node.In("NORM_RECTS"));
  } else if constexpr (std::is_same_v<TransformeeT, NormalizedRect>) {
    transformee.ConnectTo(node.In("NORM_RECT"));
  } else {
    static_assert(dependent_false<TransformeeT>::value, "Unsupported type.");
  }
  return node.Out("").template Cast<TransformeeT>();
}

}  // namespace

Stream<NormalizedRect> ScaleAndMakeSquare(
    Stream<NormalizedRect> rect, Stream<std::pair<int, int>> image_size,
    float scale_x_factor, float scale_y_factor, Graph& graph) {
  return InternalScaleAndShift(rect, image_size, scale_x_factor, scale_y_factor,
                               /*shift_x=*/std::nullopt,
                               /*shift_y=*/std::nullopt,
                               /*square_long=*/true, graph);
}

Stream<NormalizedRect> Scale(Stream<NormalizedRect> rect,
                             Stream<std::pair<int, int>> image_size,
                             float scale_x_factor, float scale_y_factor,
                             Graph& graph) {
  return InternalScaleAndShift(rect, image_size, scale_x_factor, scale_y_factor,
                               /*shift_x=*/std::nullopt,
                               /*shift_y=*/std::nullopt,
                               /*square_long=*/false, graph);
}

Stream<std::vector<NormalizedRect>> ScaleAndShiftAndMakeSquareLong(
    Stream<std::vector<NormalizedRect>> rects,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y, Graph& graph) {
  return InternalScaleAndShift(rects, image_size, scale_x_factor,
                               scale_y_factor, shift_x, shift_y,
                               /*square_long=*/true, graph);
}

Stream<std::vector<NormalizedRect>> ScaleAndShift(
    Stream<std::vector<NormalizedRect>> rects,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y, Graph& graph) {
  return InternalScaleAndShift(rects, image_size, scale_x_factor,
                               scale_y_factor, shift_x, shift_y,
                               /*square_long=*/false, graph);
}

Stream<NormalizedRect> ScaleAndShiftAndMakeSquareLong(
    Stream<NormalizedRect> rect, Stream<std::pair<int, int>> image_size,
    float scale_x_factor, float scale_y_factor, float shift_x, float shift_y,
    Graph& graph) {
  return InternalScaleAndShift(rect, image_size, scale_x_factor, scale_y_factor,
                               shift_x, shift_y,
                               /*square_long=*/true, graph);
}

Stream<NormalizedRect> ScaleAndShift(Stream<NormalizedRect> rect,
                                     Stream<std::pair<int, int>> image_size,
                                     float scale_x_factor, float scale_y_factor,
                                     float shift_x, float shift_y,
                                     Graph& graph) {
  return InternalScaleAndShift(rect, image_size, scale_x_factor, scale_y_factor,
                               shift_x, shift_y, /*square_long=*/false, graph);
}

}  // namespace mediapipe::api2::builder
