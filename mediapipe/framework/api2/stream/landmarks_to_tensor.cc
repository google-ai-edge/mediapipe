#include "mediapipe/framework/api2/stream/landmarks_to_tensor.h"

#include <optional>
#include <utility>
#include <vector>

#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe::api2::builder {

namespace {

using ::mediapipe::api2::LandmarksToTensorCalculator;

template <typename LandmarkListType>
Stream<std::vector<Tensor>> InternalConvertToTensor(
    Stream<LandmarkListType> landmarks,
    std::optional<Stream<std::pair<int, int>>> image_size,
    absl::Span<const LandmarksToTensorCalculatorOptions::Attribute> attributes,
    const bool flatten, Graph& graph) {
  auto& to_tensor = graph.AddNode<LandmarksToTensorCalculator>();
  auto& to_tensor_options =
      to_tensor.GetOptions<LandmarksToTensorCalculatorOptions>();
  for (const auto& attribute : attributes) {
    to_tensor_options.add_attributes(attribute);
  }
  to_tensor_options.set_flatten(flatten);
  if constexpr (std::is_same_v<LandmarkListType, LandmarkList>) {
    landmarks.ConnectTo(
        to_tensor[LandmarksToTensorCalculator::kInLandmarkList]);
  } else {
    landmarks.ConnectTo(
        to_tensor[LandmarksToTensorCalculator::kInNormLandmarkList]);
  }
  if (image_size.has_value()) {
    image_size->ConnectTo(to_tensor[LandmarksToTensorCalculator::kImageSize]);
  }
  return to_tensor[LandmarksToTensorCalculator::kOutTensors];
}

}  // namespace

Stream<std::vector<Tensor>> ConvertLandmarksToTensor(
    Stream<LandmarkList> landmarks,
    absl::Span<const LandmarksToTensorCalculatorOptions::Attribute> attributes,
    const bool flatten, Graph& graph) {
  return InternalConvertToTensor(landmarks, /*image_size=*/std::nullopt,
                                 attributes, flatten, graph);
}

Stream<std::vector<Tensor>> ConvertNormalizedLandmarksToTensor(
    Stream<NormalizedLandmarkList> normalized_landmarks,
    Stream<std::pair<int, int>> image_size,
    absl::Span<const LandmarksToTensorCalculatorOptions::Attribute> attributes,
    const bool flatten, Graph& graph) {
  return InternalConvertToTensor(normalized_landmarks, image_size, attributes,
                                 flatten, graph);
}

}  // namespace mediapipe::api2::builder
