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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/image_generator/image_generator_result.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {

// Options for drawing face landmarks image.
struct FaceConditionOptions {
  // The base options for plugin model.
  tasks::core::BaseOptions base_options;

  // Face landmarker options used to detect face landmarks in the condition
  // image.
  face_landmarker::FaceLandmarkerOptions face_landmarker_options;
};

// Options for detecting edges image.
struct EdgeConditionOptions {
  // The base options for plugin model.
  tasks::core::BaseOptions base_options;

  // These parameters are used to config Canny edge algorithm of OpenCV.
  // See more details:
  // https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de

  // First threshold for the hysteresis procedure.
  float threshold_1 = 100;

  // Second threshold for the hysteresis procedure.
  float threshold_2 = 200;

  // Aperture size for the Sobel operator. Typical range is 3~7.
  int aperture_size = 3;

  // A flag, indicating whether a more accurate L2 norm should be used to
  // calculate the image gradient magnitude ( L2gradient=true ), or whether
  // the default L1 norm is enough ( L2gradient=false ).
  bool l2_gradient = false;
};

// Options for detecting depth image.
struct DepthConditionOptions {
  // The base options for plugin model.
  tasks::core::BaseOptions base_options;

  // Image segmenter options used to detect depth in the condition image.
  image_segmenter::ImageSegmenterOptions image_segmenter_options;
};

struct ConditionOptions {
  enum ConditionType { FACE, EDGE, DEPTH };
  std::optional<FaceConditionOptions> face_condition_options;
  std::optional<EdgeConditionOptions> edge_condition_options;
  std::optional<DepthConditionOptions> depth_condition_options;
};

// Note: The API is experimental and subject to change.
// The options for configuring a mediapipe image generator task.
struct ImageGeneratorOptions {
  // The text to image model directory storing the model weights.
  std::string text2image_model_directory;

  enum ModelType {
    SD_1 = 1,  // Stable Diffusion v1 models, including SD 1.4 and 1.5.
  }

  model_type = ModelType::SD_1;

  // The path to LoRA weights file.
  std::optional<std::string> lora_weights_file_path;
};

class ImageGenerator : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageGenerator from the provided options.
  // image_generator_options: options to create the image generator.
  // condition_options: optional options if plugin models are used to generate
  // an image based on the condition image.
  static absl::StatusOr<std::unique_ptr<ImageGenerator>> Create(
      std::unique_ptr<ImageGeneratorOptions> image_generator_options,
      std::unique_ptr<ConditionOptions> condition_options = nullptr);

  // Create the condition image of specified condition type from the source
  // condition image. Currently support face landmarks, depth image and edge
  // image as the condition image.
  absl::StatusOr<Image> CreateConditionImage(
      Image source_condition_image,
      ConditionOptions::ConditionType condition_type);

  // Generates an image for iterations and the given random seed. Only valid
  // when the ImageGenerator is created without condition options.
  absl::StatusOr<ImageGeneratorResult> Generate(const std::string& prompt,
                                                int iterations, int seed = 0);

  // Generates an image based on the condition image for iterations and the
  // given random seed.
  // A detailed introduction to the condition image:
  // https://ai.googleblog.com/2023/06/on-device-diffusion-plugins-for.html
  absl::StatusOr<ImageGeneratorResult> Generate(
      const std::string& prompt, Image condition_image,
      ConditionOptions::ConditionType condition_type, int iterations,
      int seed = 0);

 private:
  struct ConditionInputs {
    Image condition_image;
    int select;
  };

  bool use_condition_image_ = false;

  absl::Time init_timestamp_;

  std::unique_ptr<tasks::core::TaskRunner>
      condition_image_graphs_container_task_runner_;

  std::unique_ptr<std::map<ConditionOptions::ConditionType, int>>
      condition_type_index_;

  absl::StatusOr<ImageGeneratorResult> RunIterations(
      const std::string& prompt, int steps, int rand_seed,
      std::optional<ConditionInputs> condition_inputs);
};

}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_H_
