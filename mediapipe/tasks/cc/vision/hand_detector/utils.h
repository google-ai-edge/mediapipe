#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_DETECTOR_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_DETECTOR_UTILS_H_

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "research/aimatter/api/face_detector_metadata_generated.h"

namespace mediapipe::tasks::vision::hand_detector {

// Configure SsdAnchorsCalculator from the tflite model with aimatter metadata.
absl::Status ConfigureSsdAnchorsCalculator(
    const ImageTensorSpecs& image_tensor_specs,
    const research::aimatter::api::fb::FaceDetectorMetadata& metadata_fb,
    mediapipe::SsdAnchorsCalculatorOptions& options);

// Configure TensorsToDetectionCalculator.
absl::Status ConfigureTensorsToDetectionsCalculator(
    const ImageTensorSpecs& image_tensor_specs, int num_boxes,
    float min_detection_confidence,
    mediapipe::TensorsToDetectionsCalculatorOptions& options);

}  // namespace mediapipe::tasks::vision::hand_detector

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_DETECTOR_UTILS_H_
