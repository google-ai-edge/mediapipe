#include "mediapipe/tasks/cc/vision/hand_detector/utils.h"

#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "research/aimatter/api/face_detector_metadata_generated.h"
#include "research/aimatter/api/internal/blaze_face/anchor_ssd_decoder.h"
#include "util/task/contrib/status_macros/ret_check.h"

namespace mediapipe::tasks::vision::hand_detector {

namespace rapi = ::research::aimatter::api;

constexpr int kPalmClassNum = 1;
constexpr int kBboxCoordsNum = 4;
constexpr int kPalmKeypointNum = 7;
constexpr int kKeypointCoordsNum = 2;
constexpr int kCoordsNum =
    kBboxCoordsNum + kKeypointCoordsNum * kPalmKeypointNum;

absl::Status ConfigureSsdAnchorsCalculator(
    const ImageTensorSpecs& image_tensor_specs,
    const research::aimatter::api::fb::FaceDetectorMetadata& metadata_fb,
    mediapipe::SsdAnchorsCalculatorOptions& options) {
  options.Clear();
  const auto& output_spec_fb = *metadata_fb.output_spec();
  RET_CHECK(output_spec_fb.v1() == nullptr && output_spec_fb.v2() != nullptr)
      << "Only support BlazeFaceOutputSpecV2.";
  auto* configuration = output_spec_fb.v2()->anchors_scheme()->configuration();
  std::vector<rapi::internal::AnchorSsdDecoder::AnchorConfig> configs;
  configs.reserve(configuration->Length());
  for (int i = 0; i < configuration->Length(); ++i) {
    configs.push_back({.stride = configuration->Get(i)->stride(),
                       .anchors_num = static_cast<int>(
                           configuration->Get(i)->anchors()->Length())});
  }
  const int tensor_height = image_tensor_specs.image_height;
  const int tensor_width = image_tensor_specs.image_width;
  const auto& rapi_anchors = rapi::internal::AnchorSsdDecoder::GenerateAnchors(
      configs, tensor_width, tensor_height);
  for (const auto rapi_anchor : rapi_anchors) {
    auto* anchor = options.add_fixed_anchors();
    anchor->set_x_center(rapi_anchor.center_x / tensor_width);
    anchor->set_y_center(rapi_anchor.center_y / tensor_height);
    anchor->set_w(1.0);
    anchor->set_h(1.0);
  }
  return absl::OkStatus();
}

absl::Status ConfigureTensorsToDetectionsCalculator(
    const ImageTensorSpecs& image_tensor_specs, int num_boxes,
    float min_detection_confidence,
    mediapipe::TensorsToDetectionsCalculatorOptions& options) {
  options.Clear();
  const int tensor_height = image_tensor_specs.image_height;
  const int tensor_width = image_tensor_specs.image_width;
  options.set_num_classes(kPalmClassNum);
  options.set_num_boxes(num_boxes);
  options.set_num_coords(kCoordsNum);
  options.set_box_coord_offset(0);
  options.set_keypoint_coord_offset(kBboxCoordsNum);
  options.set_num_keypoints(kPalmKeypointNum);
  options.set_num_values_per_keypoint(kKeypointCoordsNum);
  options.set_sigmoid_score(true);
  options.set_box_format(mediapipe::TensorsToDetectionsCalculatorOptions::XYWH);
  options.set_min_score_thresh(min_detection_confidence);
  options.set_x_scale(tensor_width);
  options.set_y_scale(tensor_height);
  options.set_w_scale(tensor_width);
  options.set_h_scale(tensor_height);
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::vision::hand_detector
