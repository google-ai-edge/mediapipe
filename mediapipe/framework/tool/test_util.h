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

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {
using mediapipe::CalculatorGraphConfig;

// Checks if two image frames are equal within the specified tolerance.
// image1 and image2 may be of different-but-compatible image formats (e.g.,
// SRGB and SRGBA); in that case, only the channels available in both are
// compared.
// max_color_diff applies to the first 3 channels; i.e., R, G, B for sRGB and
// sRGBA, and the single gray channel for GRAY8 and GRAY16. It is the maximum
// pixel color value difference allowed; i.e., a value from 0 to 2^8-1 for 8-bit
// data and a value from 0 to 2^16-1 for 16-bit data.
// max_alpha_diff applies to the 4th (alpha) channel only, if present.
// max_avg_diff applies to all channels, normalized across all pixels.
//
// Note: Although max_color_diff and max_alpha_diff are floats, all uint8/uint16
// values are exactly representable. (2^24 + 1 is the first non-representable
// positive integral value.)
bool CompareImageFrames(const ImageFrame& image1, const ImageFrame& image2,
                        const float max_color_diff, const float max_alpha_diff,
                        const float max_avg_diff = 1.0,
                        std::string* error_message = nullptr);

// Returns the absolute path to the directory that contains test source code.
std::string GetTestRootDir();

// Returns the absolute path to the contents of the package's "testdata"
// directory.
// This handles the different paths where test data ends up when using
// ion_cc_test on various platforms.
std::string GetTestDataDir(const std::string& package_base_path);

// Loads a binary graph from path. Returns true iff successful.
bool LoadTestGraph(CalculatorGraphConfig* proto, const std::string& path);

// Loads a PNG image from path using the given ImageFormat. Returns nullptr in
// case of failure.
std::unique_ptr<ImageFrame> LoadTestPng(
    const std::string& path, ImageFormat::Format format = ImageFormat::SRGBA);

// Returns the luminance image of |original_image|.
// The format of |original_image| must be sRGB or sRGBA.
std::unique_ptr<ImageFrame> GenerateLuminanceImage(
    const ImageFrame& original_image);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TEST_UTIL_H_
