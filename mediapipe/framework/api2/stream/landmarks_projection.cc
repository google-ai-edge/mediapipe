#include "mediapipe/framework/api2/stream/landmarks_projection.h"

#include <array>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe::api2::builder {

Stream<mediapipe::NormalizedLandmarkList> ProjectLandmarks(
    Stream<mediapipe::NormalizedLandmarkList> landmarks,
    Stream<std::array<float, 16>> projection_matrix, Graph& graph) {
  auto& projector = graph.AddNode("LandmarkProjectionCalculator");
  landmarks.ConnectTo(projector.In("NORM_LANDMARKS"));
  projection_matrix.ConnectTo(projector.In("PROJECTION_MATRIX"));
  return projector.Out("NORM_LANDMARKS")
      .Cast<mediapipe::NormalizedLandmarkList>();
}

}  // namespace mediapipe::api2::builder
