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

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#elif defined(__ANDROID__)
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

namespace {

// Checks if two values are equal within the specified tolerance.
template <typename T>
bool EqualWithTolerance(const T value1, const T value2, const T max_diff) {
  const T diff = (value1 >= value2) ? (value1 - value2) : (value2 - value1);
  return diff <= max_diff;
}

template <typename T>
bool CompareDiff(const ImageFrame& image1, const ImageFrame& image2,
                 const T max_color_diff, const T max_alpha_diff,
                 const float max_avg_diff, std::string* error_message) {
  // Verify image byte depth matches expected byte depth.
  CHECK_EQ(sizeof(T), image1.ByteDepth());
  CHECK_EQ(sizeof(T), image2.ByteDepth());

  const bool return_error = error_message != nullptr;

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

  float avg_diff = 0;
  uint diff_count = 0;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int channel = 0; channel < num_channels; ++channel) {
        // Check local difference.
        const T max_diff = channel < 3 ? max_color_diff : max_alpha_diff;
        const T value1 = pixel1[channel];
        const T value2 = pixel2[channel];
        if (!EqualWithTolerance<T>(value1, value2, max_diff)) {
          // We cast uint8 to int using this type (and leave other values as-is)
          // to avoid printing as a single char.
          using TypeToPrint =
              typename std::conditional<std::is_same<T, uint8>::value, int,
                                        T>::type;
          std::string error = absl::Substitute(
              "images differ: row = $0 col = $1 channel = $2 : pixel1 = $3, "
              "pixel2 = $4",
              row, col, channel, static_cast<TypeToPrint>(value1),
              static_cast<TypeToPrint>(value2));
          if (return_error) {
            *error_message = error;
          } else {
            LOG(ERROR) << error;
          }
          return false;
        }
        // Check global average difference.
        const float diff =
            std::abs(static_cast<float>(value1) - static_cast<float>(value2));
        avg_diff += (diff - avg_diff) / ++diff_count;
      }
      pixel1 += channels1;
      pixel2 += channels2;
    }
    pixel1 += width_padding1;
    pixel2 += width_padding2;
  }

  if (avg_diff > max_avg_diff) {
    std::string error =
        absl::Substitute("images differ: avg pixel error = $0", avg_diff);
    if (return_error) {
      *error_message = error;
    } else {
      LOG(ERROR) << error;
    }
    return false;
  }

  return true;
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

bool CompareImageFrames(const ImageFrame& image1, const ImageFrame& image2,
                        const float max_color_diff, const float max_alpha_diff,
                        const float max_avg_diff, std::string* error_message) {
  const bool return_error = error_message != nullptr;

  auto IsSupportedImageFormatComparison = [](const ImageFrame& image1,
                                             const ImageFrame& image2) {
    // Pairs of non-equal image formats that can be compared against each other.
    static const mediapipe::NoDestructor<absl::flat_hash_set<
        std::pair<ImageFormat::Format, ImageFormat::Format>>>
        kCompatibleImageFormats({
            {ImageFormat::SRGB, ImageFormat::SRGBA},
            {ImageFormat::SRGB48, ImageFormat::SRGBA64},
        });

    auto* compatible_image_formats = kCompatibleImageFormats.get();

    return image1.Format() == image2.Format() ||
           compatible_image_formats->contains(
               {image1.Format(), image2.Format()}) ||
           compatible_image_formats->contains(
               {image2.Format(), image1.Format()});
  };

  if (!IsSupportedImageFormatComparison(image1, image2)) {
    std::string error = absl::Substitute(
        "unsupported image format comparison; image1 = $0, image2 = $1",
        image1.Format(), image2.Format());
    if (return_error) {
      *error_message = error;
    } else {
      LOG(ERROR) << error;
    }
    return false;
  }

  if (image1.Width() != image2.Width()) {
    std::string error =
        absl::Substitute("image width mismatch: image1 = $0, image2 = $1",
                         image1.Width(), image2.Width());
    if (return_error) {
      *error_message = error;
    } else {
      LOG(ERROR) << error;
    }
    return false;
  }

  if (image1.Height() != image2.Height()) {
    std::string error =
        absl::Substitute("image height mismatch: image1 = $0, image2 = $1",
                         image1.Height(), image2.Height());
    if (return_error) {
      *error_message = error;
    } else {
      LOG(ERROR) << error;
    }
    return false;
  }

  if (image1.ByteDepth() != image2.ByteDepth()) {
    std::string error =
        absl::Substitute("image byte depth mismatch: image1 = $0, image2 = $1",
                         image1.ByteDepth(), image2.ByteDepth());
    if (return_error) {
      *error_message = error;
    } else {
      LOG(ERROR) << error;
    }
    return false;
  }

  switch (image1.Format()) {
    case ImageFormat::GRAY8:
    case ImageFormat::SRGB:
    case ImageFormat::SRGBA:
    case ImageFormat::LAB8:
      return CompareDiff<uint8>(image1, image2, max_color_diff, max_alpha_diff,
                                max_avg_diff, error_message);
    case ImageFormat::GRAY16:
    case ImageFormat::SRGB48:
    case ImageFormat::SRGBA64:
      return CompareDiff<uint16>(image1, image2, max_color_diff, max_alpha_diff,
                                 max_avg_diff, error_message);
    case ImageFormat::VEC32F1:
    case ImageFormat::VEC32F2:
      return CompareDiff<float>(image1, image2, max_color_diff, max_alpha_diff,
                                max_avg_diff, error_message);
    default:
      LOG(FATAL) << ImageFrame::InvalidFormatString(image1.Format());
  }
}

std::string GetTestRootDir() {
#ifdef __APPLE__
  char path[1024];
  CFURLRef bundle_url = CFBundleCopyBundleURL(CFBundleGetMainBundle());
  Boolean success = CFURLGetFileSystemRepresentation(
      bundle_url, true, reinterpret_cast<UInt8*>(path), sizeof(path));
  CHECK(success);
  CFRelease(bundle_url);
  return path;
#elif defined(__ANDROID__)
  char path[1024];
  char* ptr = getcwd(path, sizeof(path));
  CHECK_EQ(ptr, path);
  return path;
#else
  return ::mediapipe::file::JoinPath(std::getenv("TEST_SRCDIR"), "mediapipe");
#endif  // defined(__APPLE__)
}

std::string GetTestDataDir(const std::string& package_base_path) {
#ifdef __APPLE__
  return ::mediapipe::file::JoinPath(GetTestRootDir(), "testdata/");
#elif defined(__ANDROID__)
  std::string data_dir = GetTestRootDir();
  std::string binary_dir = GetBinaryDirectory();
  // In Mobile Harness, the cwd is "/" and the run dir is "/data/local/tmp".
  if (data_dir == "/" && absl::StartsWith(binary_dir, "/data")) {
    data_dir = binary_dir;
  }
  return ::mediapipe::file::JoinPath(data_dir, package_base_path, "testdata/");
#else
  return ::mediapipe::file::JoinPath(GetTestRootDir(), package_base_path,
                                     "testdata/");
#endif  // defined(__APPLE__)
}

std::unique_ptr<ImageFrame> LoadTestPng(const std::string& path,
                                        ImageFormat::Format format) {
  return nullptr;
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
