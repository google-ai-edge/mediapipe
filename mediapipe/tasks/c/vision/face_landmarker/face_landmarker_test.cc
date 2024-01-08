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

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "face_landmarker_v2_with_blendshapes.task";
constexpr char kImageFile[] = "portrait.jpg";
constexpr float kLandmarksPrecision = 0.03;
constexpr float kBlendshapesPrecision = 0.12;
constexpr float kFacialTransformationMatrixPrecision = 0.05;
constexpr int kIterations = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

void AssertFaceLandmarkerResult(const FaceLandmarkerResult* result,
                                const float blendshapes_precision,
                                const float landmark_precision,
                                const float matrix_precison) {
  // Expects to have the same number of faces detected.
  EXPECT_EQ(result->face_blendshapes_count, 1);

  // Actual blendshapes matches expected blendshapes.
  EXPECT_EQ(
      std::string{result->face_blendshapes[0].categories[0].category_name},
      "_neutral");
  EXPECT_NEAR(result->face_blendshapes[0].categories[0].score, 0.0f,
              blendshapes_precision);

  // Actual landmarks match expected landmarks.
  EXPECT_NEAR(result->face_landmarks[0].landmarks[0].x, 0.4977f,
              landmark_precision);
  EXPECT_NEAR(result->face_landmarks[0].landmarks[0].y, 0.2485f,
              landmark_precision);
  EXPECT_NEAR(result->face_landmarks[0].landmarks[0].z, -0.0305f,
              landmark_precision);

  // Expects to have at least one facial transformation matrix.
  EXPECT_GE(result->facial_transformation_matrixes_count, 1);

  // Actual matrix matches expected matrix.
  // Assuming the expected matrix is 2x2 for demonstration.
  const float expected_matrix[4] = {0.9991f, 0.0166f, -0.0374f, 0.0f};
  for (int i = 0; i < 4; ++i) {
    printf(">> %f <<", result->facial_transformation_matrixes[0].data[i]);
    EXPECT_NEAR(result->facial_transformation_matrixes[0].data[i],
                expected_matrix[i], matrix_precison);
  }
}

TEST(FaceLandmarkerTest, ImageModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  FaceLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* num_faces= */ 1,
      /* min_face_detection_confidence= */ 0.5,
      /* min_face_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
      /* output_face_blendshapes = */ true,
      /* output_facial_transformation_matrixes = */ true,
  };

  void* landmarker = face_landmarker_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(landmarker, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  FaceLandmarkerResult result;
  face_landmarker_detect_image(landmarker, mp_image, &result,
                               /* error_msg */ nullptr);
  AssertFaceLandmarkerResult(&result, kBlendshapesPrecision,
                             kLandmarksPrecision,
                             kFacialTransformationMatrixPrecision);
  face_landmarker_close_result(&result);
  face_landmarker_close(landmarker, /* error_msg */ nullptr);
}

TEST(FaceLandmarkerTest, VideoModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  FaceLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* num_faces= */ 1,
      /* min_face_detection_confidence= */ 0.5,
      /* min_face_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
      /* output_face_blendshapes = */ true,
      /* output_facial_transformation_matrixes = */ true,
  };

  void* landmarker = face_landmarker_create(&options,
                                            /* error_msg */ nullptr);
  EXPECT_NE(landmarker, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    FaceLandmarkerResult result;
    face_landmarker_detect_for_video(landmarker, mp_image, i, &result,
                                     /* error_msg */ nullptr);

    AssertFaceLandmarkerResult(&result, kBlendshapesPrecision,
                               kLandmarksPrecision,
                               kFacialTransformationMatrixPrecision);
    face_landmarker_close_result(&result);
  }
  face_landmarker_close(landmarker, /* error_msg */ nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static void Fn(const FaceLandmarkerResult* landmarker_result,
                 const MpImage& image, int64_t timestamp, char* error_msg) {
    ASSERT_NE(landmarker_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    AssertFaceLandmarkerResult(landmarker_result, kBlendshapesPrecision,
                               kLandmarksPrecision,
                               kFacialTransformationMatrixPrecision);
    EXPECT_GT(image.image_frame.width, 0);
    EXPECT_GT(image.image_frame.height, 0);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

TEST(FaceLandmarkerTest, LiveStreamModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);

  FaceLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* num_faces= */ 1,
      /* min_face_detection_confidence= */ 0.5,
      /* min_face_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
      /* output_face_blendshapes = */ true,
      /* output_facial_transformation_matrixes = */ true,
      /* result_callback= */ LiveStreamModeCallback::Fn,
  };

  void* landmarker = face_landmarker_create(&options, /* error_msg */
                                            nullptr);
  EXPECT_NE(landmarker, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(face_landmarker_detect_async(landmarker, mp_image, i,
                                           /* error_msg */ nullptr),
              0);
  }
  face_landmarker_close(landmarker, /* error_msg */ nullptr);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(FaceLandmarkerTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  FaceLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* running_mode= */ RunningMode::IMAGE,
      /* num_faces= */ 1,
      /* min_face_detection_confidence= */ 0.5,
      /* min_face_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
      /* output_face_blendshapes = */ true,
      /* output_facial_transformation_matrixes = */ true,
  };

  char* error_msg;
  void* landmarker = face_landmarker_create(&options, &error_msg);
  EXPECT_EQ(landmarker, nullptr);

  EXPECT_THAT(
      error_msg,
      HasSubstr("INVALID_ARGUMENT: BLENDSHAPES Tag and blendshapes model must "
                "be both set. Get BLENDSHAPES is set: true, blendshapes model "
                "is set: false [MediaPipeTasksStatus='601']"));

  free(error_msg);
}

TEST(FaceLandmarkerTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  FaceLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* num_faces= */ 1,
      /* min_face_detection_confidence= */ 0.5,
      /* min_face_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
      /* output_face_blendshapes = */ true,
      /* output_facial_transformation_matrixes = */ true,
  };

  void* landmarker = face_landmarker_create(&options, /* error_msg */
                                            nullptr);
  EXPECT_NE(landmarker, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  FaceLandmarkerResult result;
  char* error_msg;
  face_landmarker_detect_image(landmarker, mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  face_landmarker_close(landmarker, /* error_msg */ nullptr);
}

}  // namespace
