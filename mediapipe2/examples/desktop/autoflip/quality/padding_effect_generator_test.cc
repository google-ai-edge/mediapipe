// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/examples/desktop/autoflip/quality/padding_effect_generator.h"

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/status_matchers.h"

ABSL_FLAG(std::string, input_image, "", "The path to an input image.");
ABSL_FLAG(std::string, output_folder, "",
          "The folder to output test result images.");

namespace mediapipe {
namespace autoflip {
namespace {

// An 320x180 RGB test image.
constexpr char kTestImage[] =
    "mediapipe/examples/desktop/autoflip/quality/testdata/"
    "google.jpg";
constexpr char kResultImagePrefix[] =
    "mediapipe/examples/desktop/autoflip/quality/testdata/"
    "result_";

const cv::Scalar kRed = cv::Scalar(255, 0, 0);

void TestWithAspectRatio(const double aspect_ratio,
                         const cv::Scalar* background_color_in_rgb = nullptr) {
  std::string test_image;
  const bool process_arbitrary_image =
      !absl::GetFlag(FLAGS_input_image).empty();
  if (!process_arbitrary_image) {
    std::string test_image_path = mediapipe::file::JoinPath("./", kTestImage);
    MP_ASSERT_OK(mediapipe::file::GetContents(test_image_path, &test_image));
  } else {
    MP_ASSERT_OK(mediapipe::file::GetContents(absl::GetFlag(FLAGS_input_image),
                                              &test_image));
  }

  const std::vector<char> contents_vector(test_image.begin(), test_image.end());
  cv::Mat decoded_mat =
      cv::imdecode(contents_vector, -1 /* return the loaded image as-is */);

  ImageFormat::Format image_format = ImageFormat::UNKNOWN;
  cv::Mat output_mat;
  switch (decoded_mat.channels()) {
    case 1:
      image_format = ImageFormat::GRAY8;
      output_mat = decoded_mat;
      break;
    case 3:
      image_format = ImageFormat::SRGB;
      cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGB);
      break;
    case 4:
      MP_ASSERT_OK(mediapipe::UnimplementedErrorBuilder(MEDIAPIPE_LOC)
                   << "4-channel image isn't supported yet");
      break;
    default:
      MP_ASSERT_OK(mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
                   << "Unsupported number of channels: "
                   << decoded_mat.channels());
  }
  std::unique_ptr<ImageFrame> test_frame = absl::make_unique<ImageFrame>(
      image_format, decoded_mat.size().width, decoded_mat.size().height);
  output_mat.copyTo(formats::MatView(test_frame.get()));

  PaddingEffectGenerator generator(test_frame->Width(), test_frame->Height(),
                                   aspect_ratio);
  ImageFrame result_frame;
  MP_ASSERT_OK(generator.Process(*test_frame, 0.3, 40, 0.0, &result_frame,
                                 background_color_in_rgb));
  cv::Mat original_mat = formats::MatView(&result_frame);
  cv::Mat input_mat;
  switch (original_mat.channels()) {
    case 1:
      input_mat = original_mat;
      break;
    case 3:
      // OpenCV assumes the image to be BGR order. To use imencode(), do color
      // conversion first.
      cv::cvtColor(original_mat, input_mat, cv::COLOR_RGB2BGR);
      break;
    case 4:
      MP_ASSERT_OK(mediapipe::UnimplementedErrorBuilder(MEDIAPIPE_LOC)
                   << "4-channel image isn't supported yet");
      break;
    default:
      MP_ASSERT_OK(mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
                   << "Unsupported number of channels: "
                   << original_mat.channels());
  }

  std::vector<int> parameters;
  parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
  constexpr int kEncodingQuality = 75;
  parameters.push_back(kEncodingQuality);

  std::vector<uchar> encode_buffer;
  // Note that imencode() will store the data in RGB order.
  // Check its JpegEncoder::write() in "imgcodecs/src/grfmt_jpeg.cpp" for more
  // info.
  if (!cv::imencode(".jpg", input_mat, encode_buffer, parameters)) {
    MP_ASSERT_OK(mediapipe::InternalErrorBuilder(MEDIAPIPE_LOC)
                 << "Fail to encode the image to be jpeg format.");
  }

  std::string output_string(absl::string_view(
      reinterpret_cast<const char*>(&encode_buffer[0]), encode_buffer.size()));

  if (!process_arbitrary_image) {
    std::string result_string_path = mediapipe::file::JoinPath(
        "./", absl::StrCat(kResultImagePrefix, aspect_ratio,
                           background_color_in_rgb ? "_solid_background" : "",
                           ".jpg"));
    std::string result_image;
    MP_ASSERT_OK(
        mediapipe::file::GetContents(result_string_path, &result_image));
    EXPECT_EQ(result_image, output_string);
  } else {
    std::string output_string_path = mediapipe::file::JoinPath(
        absl::GetFlag(FLAGS_output_folder),
        absl::StrCat("result_", aspect_ratio,
                     background_color_in_rgb ? "_solid_background" : "",
                     ".jpg"));
    MP_ASSERT_OK(
        mediapipe::file::SetContents(output_string_path, output_string));
  }
}

TEST(PaddingEffectGeneratorTest, Success) {
  TestWithAspectRatio(0.3);
  TestWithAspectRatio(0.6);
  TestWithAspectRatio(1.0);
  TestWithAspectRatio(1.6);
  TestWithAspectRatio(2.5);
  TestWithAspectRatio(3.4);
}

TEST(PaddingEffectGeneratorTest, SuccessWithBackgroundColor) {
  TestWithAspectRatio(0.3, &kRed);
  TestWithAspectRatio(0.6, &kRed);
  TestWithAspectRatio(1.0, &kRed);
  TestWithAspectRatio(1.6, &kRed);
  TestWithAspectRatio(2.5, &kRed);
  TestWithAspectRatio(3.4, &kRed);
}

TEST(PaddingEffectGeneratorTest, ScaleToMultipleOfTwo) {
  int input_width = 30;
  int input_height = 30;
  double target_aspect_ratio = 0.5;
  int expect_width = 14;
  int expect_height = input_height;
  auto test_frame = absl::make_unique<ImageFrame>(/*format=*/ImageFormat::SRGB,
                                                  input_width, input_height);

  PaddingEffectGenerator generator(test_frame->Width(), test_frame->Height(),
                                   target_aspect_ratio,
                                   /*scale_to_multiple_of_two=*/true);
  ImageFrame result_frame;
  MP_ASSERT_OK(generator.Process(*test_frame, 0.3, 40, 0.0, &result_frame));
  EXPECT_EQ(result_frame.Width(), expect_width);
  EXPECT_EQ(result_frame.Height(), expect_height);
}

TEST(PaddingEffectGeneratorTest, ComputeOutputLocation) {
  PaddingEffectGenerator generator(1920, 1080, 1.0);

  auto result_rect = generator.ComputeOutputLocation();
  EXPECT_EQ(result_rect.x, 0);
  EXPECT_EQ(result_rect.y, 236);
  EXPECT_EQ(result_rect.width, 1080);
  EXPECT_EQ(result_rect.height, 607);
}
}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
