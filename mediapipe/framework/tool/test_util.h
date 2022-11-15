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

#ifndef MEDIAPIPE_FRAMEWORK_TEST_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_TEST_UTIL_H_

#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {
using mediapipe::CalculatorGraphConfig;

struct ImageFrameComparisonOptions {
  // NOTE: these values are not normalized: use a value from 0 to 2^8-1
  // for 8-bit data and a value from 0 to 2^16-1 for 16-bit data.
  // Although these members are declared as floats,, all uint8/uint16
  // values are exactly representable. (2^24 + 1 is the first non-representable
  // positive integral value.)

  // Maximum value difference allowed for non-alpha channels.
  float max_color_diff;
  // Maximum value difference allowed for alpha channel (if present).
  float max_alpha_diff;
  // Maximum difference for all channels, averaged across all pixels.
  float max_avg_diff;
};

// Compares an output image with a golden file. Saves the output and difference
// to the undeclared test outputs.
// Returns ok if they are equal within the tolerances specified in options.
absl::Status CompareAndSaveImageOutput(
    absl::string_view golden_image_path, const ImageFrame& actual,
    const ImageFrameComparisonOptions& options);

// Checks if two image frames are equal within the specified tolerance.
// image1 and image2 may be of different-but-compatible image formats (e.g.,
// SRGB and SRGBA); in that case, only the channels available in both are
// compared.
// The diff arguments are as in ImageFrameComparisonOptions.
absl::Status CompareImageFrames(const ImageFrame& image1,
                                const ImageFrame& image2,
                                const float max_color_diff,
                                const float max_alpha_diff,
                                const float max_avg_diff,
                                std::unique_ptr<ImageFrame>& diff_image);

bool CompareImageFrames(const ImageFrame& image1, const ImageFrame& image2,
                        const float max_color_diff, const float max_alpha_diff,
                        const float max_avg_diff = 1.0,
                        std::string* error_message = nullptr);

// Returns the absolute path to the directory that contains test source code
// (TEST_SRCDIR).
std::string GetTestRootDir();

// Returns the absolute path to a directory where tests can write outputs to
// be sent to bazel (TEST_UNDECLARED_OUTPUTS_DIR or a fallback).
std::string GetTestOutputsDir();

// Returns the absolute path to a file within TEST_SRCDIR.
std::string GetTestFilePath(absl::string_view relative_path);

// Returns the absolute path to the contents of the package's "testdata"
// directory.
// This handles the different paths where test data ends up when using
// ion_cc_test on various platforms.
std::string GetTestDataDir(absl::string_view package_base_path);

// Loads a binary graph from path. Returns true iff successful.
bool LoadTestGraph(CalculatorGraphConfig* proto, const std::string& path);

// Loads an image from memory.
absl::StatusOr<std::unique_ptr<ImageFrame>> DecodeTestImage(
    absl::string_view encoded, ImageFormat::Format format = ImageFormat::SRGBA);

// Loads an image from path.
absl::StatusOr<std::unique_ptr<ImageFrame>> LoadTestImage(
    absl::string_view path, ImageFormat::Format format = ImageFormat::SRGBA);

// Loads a PNG image from path using the given ImageFormat. Returns nullptr in
// case of failure.
std::unique_ptr<ImageFrame> LoadTestPng(
    absl::string_view path, ImageFormat::Format format = ImageFormat::SRGBA);

// Write an ImageFrame as PNG to the test undeclared outputs directory.
// The image's name will contain the given prefix and a timestamp.
// If successful, returns the path to the output file relative to the output
// directory.
absl::StatusOr<std::string> SavePngTestOutput(
    const mediapipe::ImageFrame& image, absl::string_view prefix);

// Returns the luminance image of |original_image|.
// The format of |original_image| must be sRGB or sRGBA.
std::unique_ptr<ImageFrame> GenerateLuminanceImage(
    const ImageFrame& original_image);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TEST_UTIL_H_
