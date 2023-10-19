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

#include "mediapipe/tasks/c/vision/image_classifier/image_classifier.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "mobilenet_v2_1.0_224.tflite";
constexpr float kPrecision = 1e-4;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(ImageClassifierTest, SmokeTest) {
  const auto image = DecodeImageFromFile(GetFullPath("burger.jpg"));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ -1,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
  };

  void* classifier = image_classifier_create(&options);
  EXPECT_NE(classifier, nullptr);

  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {
          .format = static_cast<ImageFormat>(
              image->GetImageFrameSharedPtr()->Format()),
          .image_buffer = image->GetImageFrameSharedPtr()->PixelData(),
          .width = image->GetImageFrameSharedPtr()->Width(),
          .height = image->GetImageFrameSharedPtr()->Height()}};

  ImageClassifierResult result;
  image_classifier_classify_image(classifier, &mp_image, &result);
  EXPECT_EQ(result.classifications_count, 1);
  EXPECT_EQ(result.classifications[0].categories_count, 1001);
  EXPECT_EQ(std::string{result.classifications[0].categories[0].category_name},
            "cheeseburger");
  EXPECT_NEAR(result.classifications[0].categories[0].score, 0.7939f,
              kPrecision);
  image_classifier_close_result(&result);
  image_classifier_close(classifier);
}

TEST(ImageClassifierTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_path= */ nullptr},
      /* classifier_options= */ {},
  };

  char* error_msg;
  void* classifier = image_classifier_create(&options, &error_msg);
  EXPECT_EQ(classifier, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(ImageClassifierTest, FailedClassificationHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ -1,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
  };

  void* classifier = image_classifier_create(&options);
  EXPECT_NE(classifier, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  ImageClassifierResult result;
  char* error_msg;
  image_classifier_classify_image(classifier, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("gpu buffer not supported yet"));
  free(error_msg);
  image_classifier_close(classifier);
}

}  // namespace
