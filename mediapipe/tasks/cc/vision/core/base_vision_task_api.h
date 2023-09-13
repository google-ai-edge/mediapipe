/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_CORE_BASE_VISION_TASK_API_H_
#define MEDIAPIPE_TASKS_CC_VISION_CORE_BASE_VISION_TASK_API_H_

#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace core {

// The base class of the user-facing mediapipe vision task api classes.
class BaseVisionTaskApi : public tasks::core::BaseTaskApi {
 public:
  // Constructor.
  explicit BaseVisionTaskApi(std::unique_ptr<tasks::core::TaskRunner> runner,
                             RunningMode running_mode)
      : BaseTaskApi(std::move(runner)), running_mode_(running_mode) {}
  virtual ~BaseVisionTaskApi() {}

  virtual absl::StatusOr<ImageTensorSpecs> GetInputImageTensorSpecs() {
    ImageTensorSpecs image_tensor_specs;
    bool found_image_to_tensor_calculator = false;
    for (auto& node : runner_->GetGraphConfig().node()) {
      if (node.calculator() == "ImageToTensorCalculator") {
        if (!found_image_to_tensor_calculator) {
          found_image_to_tensor_calculator = true;
        } else {
          return absl::Status(CreateStatusWithPayload(
              absl::StatusCode::kFailedPrecondition,
              absl::StrCat(
                  "The graph has more than one ImageToTensorCalculator.")));
        }
        mediapipe::ImageToTensorCalculatorOptions options =
            node.options().GetExtension(
                mediapipe::ImageToTensorCalculatorOptions::ext);
        image_tensor_specs.image_width = options.output_tensor_width();
        image_tensor_specs.image_height = options.output_tensor_height();
        image_tensor_specs.color_space =
            tflite::ColorSpaceType::ColorSpaceType_RGB;
        if (options.has_output_tensor_uint_range()) {
          image_tensor_specs.tensor_type = tflite::TensorType_UINT8;
        } else if (options.has_output_tensor_float_range()) {
          image_tensor_specs.tensor_type = tflite::TensorType_FLOAT32;
        }
      }
    }
    if (!found_image_to_tensor_calculator) {
      return absl::Status(CreateStatusWithPayload(
          absl::StatusCode::kNotFound,
          absl::StrCat("The graph doesn't contain ImageToTensorCalculator.")));
    }
    return image_tensor_specs;
  }

 protected:
  // A synchronous method to process single image inputs.
  // The call blocks the current thread until a failure status or a successful
  // result is returned.
  absl::StatusOr<tasks::core::PacketMap> ProcessImageData(
      tasks::core::PacketMap inputs) {
    if (running_mode_ != RunningMode::IMAGE) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Task is not initialized with the image mode. Current "
                       "running mode:",
                       GetRunningModeName(running_mode_)),
          MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
    }
    return runner_->Process(std::move(inputs));
  }

  // A synchronous method to process continuous video frames.
  // The call blocks the current thread until a failure status or a successful
  // result is returned.
  absl::StatusOr<tasks::core::PacketMap> ProcessVideoData(
      tasks::core::PacketMap inputs) {
    if (running_mode_ != RunningMode::VIDEO) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Task is not initialized with the video mode. Current "
                       "running mode:",
                       GetRunningModeName(running_mode_)),
          MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
    }
    return runner_->Process(std::move(inputs));
  }

  // An asynchronous method to send live stream data to the runner. The results
  // will be available in the user-defined results callback.
  absl::Status SendLiveStreamData(tasks::core::PacketMap inputs) {
    if (running_mode_ != RunningMode::LIVE_STREAM) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Task is not initialized with the live stream mode. "
                       "Current running mode:",
                       GetRunningModeName(running_mode_)),
          MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
    }
    return runner_->Send(std::move(inputs));
  }

  // Convert from ImageProcessingOptions to NormalizedRect, performing sanity
  // checks on-the-fly. If the input ImageProcessingOptions is not present,
  // returns a default NormalizedRect covering the whole image with rotation set
  // to 0. If 'roi_allowed' is false, an error will be returned if the input
  // ImageProcessingOptions has its 'region_or_interest' field set.
  static absl::StatusOr<mediapipe::NormalizedRect> ConvertToNormalizedRect(
      std::optional<ImageProcessingOptions> options,
      const mediapipe::Image& image, bool roi_allowed = true) {
    mediapipe::NormalizedRect normalized_rect;
    normalized_rect.set_rotation(0);
    normalized_rect.set_x_center(0.5);
    normalized_rect.set_y_center(0.5);
    normalized_rect.set_width(1.0);
    normalized_rect.set_height(1.0);
    if (!options.has_value()) {
      return normalized_rect;
    }

    if (options->rotation_degrees % 90 != 0) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Expected rotation to be a multiple of 90°.",
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
    }
    // Convert to radians counter-clockwise.
    normalized_rect.set_rotation(-options->rotation_degrees * M_PI / 180.0);

    if (options->region_of_interest.has_value()) {
      if (!roi_allowed) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "This task doesn't support region-of-interest.",
            MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
      }
      auto& roi = *options->region_of_interest;
      if (roi.left >= roi.right || roi.top >= roi.bottom) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "Expected RectF with left < right and top < bottom.",
            MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
      }
      if (roi.left < 0 || roi.top < 0 || roi.right > 1 || roi.bottom > 1) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "Expected RectF values to be in [0,1].",
            MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
      }
      normalized_rect.set_x_center((roi.left + roi.right) / 2.0);
      normalized_rect.set_y_center((roi.top + roi.bottom) / 2.0);
      normalized_rect.set_width(roi.right - roi.left);
      normalized_rect.set_height(roi.bottom - roi.top);
    }

    // For 90° and 270° rotations, we need to swap width and height.
    // This is due to the internal behavior of ImageToTensorCalculator, which:
    // - first denormalizes the provided rect by multiplying the rect width or
    //   height by the image width or height, respectively.
    // - then rotates this by denormalized rect by the provided rotation, and
    //   uses this for cropping,
    // - then finally rotates this back.
    if (std::abs(options->rotation_degrees) % 180 != 0) {
      float w = normalized_rect.height() * image.height() / image.width();
      float h = normalized_rect.width() * image.width() / image.height();
      normalized_rect.set_width(w);
      normalized_rect.set_height(h);
    }

    return normalized_rect;
  }

 private:
  RunningMode running_mode_;
};

}  // namespace core
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_CORE_BASE_VISION_TASK_API_H_
