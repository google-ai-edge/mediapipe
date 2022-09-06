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

#include "mediapipe/framework/tool/test_util.h"

#include <fcntl.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace mediapipe {

namespace {

// Checks if two values are equal within the specified tolerance.
template <typename T>
bool EqualWithTolerance(const T value1, const T value2, const T max_diff) {
  const T diff = (value1 >= value2) ? (value1 - value2) : (value2 - value1);
  return diff <= max_diff;
}

template <typename T>
absl::Status CompareDiff(const ImageFrame& image1, const ImageFrame& image2,
                         const T max_color_diff, const T max_alpha_diff,
                         const float max_avg_diff,
                         std::unique_ptr<ImageFrame>& diff_image) {
  // Verify image byte depth matches expected byte depth.
  CHECK_EQ(sizeof(T), image1.ByteDepth());
  CHECK_EQ(sizeof(T), image2.ByteDepth());

  const int width = image1.Width();
  const int height = image1.Height();
  const int channels1 = image1.NumberOfChannels();
  const int channels2 = image2.NumberOfChannels();
  const T* pixel1 = reinterpret_cast<const T*>(image1.PixelData());
  const T* pixel2 = reinterpret_cast<const T*>(image2.PixelData());
  const int num_channels = std::min(channels1, channels2);

  // Verify the width steps are multiples of byte depth.
  CHECK_EQ(image1.WidthStep() % image1.ByteDepth(), 0);
  CHECK_EQ(image2.WidthStep() % image2.ByteDepth(), 0);
  const int width_padding1 =
      image1.WidthStep() / image1.ByteDepth() - width * channels1;
  const int width_padding2 =
      image2.WidthStep() / image2.ByteDepth() - width * channels2;

  diff_image = std::make_unique<ImageFrame>(image1.Format(), width, height);
  T* pixel_diff = reinterpret_cast<T*>(diff_image->MutablePixelData());
  const int width_padding_diff =
      diff_image->WidthStep() / diff_image->ByteDepth() - width * channels1;

  float avg_diff = 0;
  uint total_count = 0;
  int different_color_components = 0;
  float max_color_diff_found = 0;
  int different_alpha_components = 0;
  float max_alpha_diff_found = 0;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int channel = 0; channel < num_channels; ++channel) {
        // Check local difference.
        const T value1 = pixel1[channel];
        const T value2 = pixel2[channel];
        const float diff =
            std::abs(static_cast<float>(value1) - static_cast<float>(value2));
        if (channel < 3) {
          different_color_components += diff > max_color_diff;
          max_color_diff_found = std::max(max_color_diff_found, diff);
          pixel_diff[channel] = diff;
        } else {
          different_alpha_components += diff > max_alpha_diff;
          max_alpha_diff_found = std::max(max_alpha_diff_found, diff);
          pixel_diff[channel] = 255;  // opaque to see color difference
        }
        // Check global average difference.
        avg_diff += (diff - avg_diff) / ++total_count;
      }
      pixel1 += channels1;
      pixel2 += channels2;
      pixel_diff += channels1;
    }
    pixel1 += width_padding1;
    pixel2 += width_padding2;
    pixel_diff += width_padding_diff;
  }

  std::vector<std::string> errors;
  if (different_color_components)
    errors.push_back(absl::Substitute(
        "$0 color components differences above limit of $1, max found was $2",
        different_color_components, max_color_diff, max_color_diff_found));
  if (different_alpha_components)
    errors.push_back(absl::Substitute(
        "$0 alpha components differences above limit of $1, max found was $2",
        different_alpha_components, max_alpha_diff, max_alpha_diff_found));
  if (avg_diff > max_avg_diff)
    errors.push_back(
        absl::Substitute("the average component difference is $0 (limit: $1)",
                         avg_diff, max_avg_diff));

  if (!errors.empty())
    return absl::InternalError(
        absl::StrCat("images differ: ", absl::StrJoin(errors, "; ")));
  return absl::OkStatus();
}

#if defined(__linux__)
// Returns the directory of the running test binary.
std::string GetBinaryDirectory() {
  char full_path[PATH_MAX + 1];
  int length = readlink("/proc/self/exe", full_path, PATH_MAX + 1);
  CHECK_GT(length, 0);
  return std::string(
      ::mediapipe::file::Dirname(absl::string_view(full_path, length)));
}
#endif

}  // namespace

absl::Status CompareImageFrames(const ImageFrame& image1,
                                const ImageFrame& image2,
                                const float max_color_diff,
                                const float max_alpha_diff,
                                const float max_avg_diff,
                                std::unique_ptr<ImageFrame>& diff_image) {
  auto IsSupportedImageFormatComparison = [](ImageFormat::Format one,
                                             ImageFormat::Format two) {
    auto both = std::minmax(one, two);
    return one == two ||
           both == std::minmax(ImageFormat::SRGB, ImageFormat::SRGBA) ||
           both == std::minmax(ImageFormat::SRGB48, ImageFormat::SRGBA64);
  };

  RET_CHECK(IsSupportedImageFormatComparison(image1.Format(), image2.Format()))
      << "unsupported image format comparison; image1 = " << image1.Format()
      << ", image2 = " << image2.Format();

  // Cannot use RET_CHECK_EQ because pair is not printable.
  RET_CHECK(std::make_pair(image1.Width(), image1.Height()) ==
            std::make_pair(image2.Width(), image2.Height()))
      << "image size mismatch: " << image1.Width() << "x" << image1.Height()
      << " != " << image2.Width() << "x" << image2.Height();

  RET_CHECK_EQ(image1.ByteDepth(), image2.ByteDepth())
      << "image byte depth mismatch";

  switch (image1.Format()) {
    case ImageFormat::GRAY8:
    case ImageFormat::SRGB:
    case ImageFormat::SRGBA:
    case ImageFormat::LAB8:
      return CompareDiff<uint8>(image1, image2, max_color_diff, max_alpha_diff,
                                max_avg_diff, diff_image);
    case ImageFormat::GRAY16:
    case ImageFormat::SRGB48:
    case ImageFormat::SRGBA64:
      return CompareDiff<uint16>(image1, image2, max_color_diff, max_alpha_diff,
                                 max_avg_diff, diff_image);
    case ImageFormat::VEC32F1:
    case ImageFormat::VEC32F2:
      return CompareDiff<float>(image1, image2, max_color_diff, max_alpha_diff,
                                max_avg_diff, diff_image);
    default:
      LOG(FATAL) << ImageFrame::InvalidFormatString(image1.Format());
  }
}

bool CompareImageFrames(const ImageFrame& image1, const ImageFrame& image2,
                        const float max_color_diff, const float max_alpha_diff,
                        const float max_avg_diff, std::string* error_message) {
  std::unique_ptr<ImageFrame> diff_image;
  auto status = CompareImageFrames(image1, image2, max_color_diff,
                                   max_alpha_diff, max_avg_diff, diff_image);
  if (status.ok()) return true;
  if (error_message) *error_message = std::string(status.message());
  return false;
}

absl::Status CompareAndSaveImageOutput(
    absl::string_view golden_image_path, const ImageFrame& actual,
    const ImageFrameComparisonOptions& options) {
  ASSIGN_OR_RETURN(auto output_img_path, SavePngTestOutput(actual, "output"));

  auto expected = LoadTestImage(GetTestFilePath(golden_image_path));
  if (!expected.ok()) {
    return expected.status();
  }
  ASSIGN_OR_RETURN(auto expected_img_path,
                   SavePngTestOutput(**expected, "expected"));

  std::unique_ptr<ImageFrame> diff_img;
  auto status = CompareImageFrames(**expected, actual, options.max_color_diff,
                                   options.max_alpha_diff, options.max_avg_diff,
                                   diff_img);
  ASSIGN_OR_RETURN(auto diff_img_path, SavePngTestOutput(*diff_img, "diff"));

  return status;
}

std::string GetTestRootDir() {
  return file::JoinPath(std::getenv("TEST_SRCDIR"), "mediapipe");
}

std::string GetTestOutputsDir() {
  const char* output_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (!output_dir) {
#ifdef __APPLE__
    char path[PATH_MAX];
    size_t n = confstr(_CS_DARWIN_USER_TEMP_DIR, path, sizeof(path));
    if (n > 0 && n < sizeof(path)) return path;
#endif  // __APPLE__
#ifdef __ANDROID__
    return "/data/local/tmp/";
#endif  // __ANDROID__
    output_dir = "/tmp";
  }
  return output_dir;
}

std::string GetTestDataDir(absl::string_view package_base_path) {
  return file::JoinPath(GetTestRootDir(), package_base_path, "testdata/");
}

std::string GetTestFilePath(absl::string_view relative_path) {
  return file::JoinPath(GetTestRootDir(), relative_path);
}

absl::StatusOr<std::unique_ptr<ImageFrame>> LoadTestImage(
    absl::string_view path, ImageFormat::Format format) {
  std::string encoded;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(path, &encoded));

  // stbi_load determines the output pixel format based on the desired channels.
  // 0 means "use whatever's in the file".
  int desired_channels = format == ImageFormat::UNKNOWN ? 0
                         : format == ImageFormat::SRGBA ? 4
                         : format == ImageFormat::SRGB  ? 3
                         : format == ImageFormat::GRAY8 ? 1
                                                        : -1;
  RET_CHECK(desired_channels >= 0)
      << "unsupported output format requested: " << format;

  int width, height, channels_in_file;
  auto data = stbi_load_from_memory(reinterpret_cast<stbi_uc*>(encoded.data()),
                                    encoded.size(), &width, &height,
                                    &channels_in_file, desired_channels);
  RET_CHECK(data) << "failed to decode image data from: " << path;

  // If we didn't specify a desired format, it will be determined by what the
  // file contains.
  int output_channels = desired_channels ? desired_channels : channels_in_file;
  if (format == ImageFormat::UNKNOWN) {
    format = output_channels == 4   ? ImageFormat::SRGBA
             : output_channels == 3 ? ImageFormat::SRGB
             : output_channels == 1 ? ImageFormat::GRAY8
                                    : ImageFormat::UNKNOWN;
    RET_CHECK(format != ImageFormat::UNKNOWN)
        << "unsupported number of channels: " << output_channels;
  }

  return absl::make_unique<ImageFrame>(
      format, width, height, width * output_channels, data, stbi_image_free);
}

std::unique_ptr<ImageFrame> LoadTestPng(absl::string_view path,
                                        ImageFormat::Format format) {
  return nullptr;
}

// Write an ImageFrame as PNG to the test undeclared outputs directory.
// The image's name will contain the given prefix and a timestamp.
// Returns the path to the output if successful.
absl::StatusOr<std::string> SavePngTestOutput(
    const mediapipe::ImageFrame& image, absl::string_view prefix) {
  std::string now_string = absl::FormatTime(absl::Now());
  std::string output_relative_path =
      absl::StrCat(prefix, "_", now_string, ".png");
  std::string output_full_path =
      file::JoinPath(GetTestOutputsDir(), output_relative_path);
  RET_CHECK(stbi_write_png(output_full_path.c_str(), image.Width(),
                           image.Height(), image.NumberOfChannels(),
                           image.PixelData(), image.WidthStep()))
      << " path: " << output_full_path;
  return output_relative_path;
}

bool LoadTestGraph(CalculatorGraphConfig* proto, const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    LOG(ERROR) << "could not open test graph: " << path
               << ", error: " << strerror(errno);
    return false;
  }
  proto_ns::io::FileInputStream input(fd);
  bool success = proto->ParseFromZeroCopyStream(&input);
  close(fd);
  if (!success) {
    LOG(ERROR) << "could not parse test graph: " << path;
  }
  return success;
}

std::unique_ptr<ImageFrame> GenerateLuminanceImage(
    const ImageFrame& original_image) {
  const int width = original_image.Width();
  const int height = original_image.Height();
  const int channels = original_image.NumberOfChannels();
  if (channels != 3 && channels != 4) {
    LOG(ERROR) << "Invalid number of image channels: " << channels;
    return nullptr;
  }
  auto luminance_image =
      absl::make_unique<ImageFrame>(original_image.Format(), width, height,
                                    ImageFrame::kGlDefaultAlignmentBoundary);
  const uint8* pixel1 = original_image.PixelData();
  uint8* pixel2 = luminance_image->MutablePixelData();
  const int width_padding1 = original_image.WidthStep() - width * channels;
  const int width_padding2 = luminance_image->WidthStep() - width * channels;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      float luminance =
          pixel1[0] * 0.2125f + pixel1[1] * 0.7154f + pixel1[2] * 0.0721f;
      uint8 luminance_byte = 255;
      if (luminance < 255.0f) {
        luminance_byte = static_cast<uint8>(luminance);
      }
      pixel2[0] = luminance_byte;
      pixel2[1] = luminance_byte;
      pixel2[2] = luminance_byte;
      if (channels == 4) {
        pixel2[3] = pixel1[3];
      }
      pixel1 += channels;
      pixel2 += channels;
    }
    pixel1 += width_padding1;
    pixel2 += width_padding2;
  }
  return luminance_image;
}

}  // namespace mediapipe
