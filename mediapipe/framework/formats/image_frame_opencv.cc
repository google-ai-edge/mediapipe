// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/image_frame_opencv.h"

#include "mediapipe/framework/formats/image_format.pb.h"

namespace {
// Maps ImageFrame format to OpenCV Mat type.
// See mediapipe...image_format.proto and cv...opencv2/core/hal/interface.h
// for more details on respective formats.
int GetMatType(const mediapipe::ImageFormat::Format format) {
  int type = 0;
  switch (format) {
    case mediapipe::ImageFormat::UNKNOWN:
      // Invalid; Default to uchar.
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SRGB:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SRGBA:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::GRAY8:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::GRAY16:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::YCBCR420P:
      // Invalid; Default to uchar.
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::YCBCR420P10:
      // Invalid; Default to uint16.
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::SRGB48:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::SRGBA64:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::VEC32F1:
      type = CV_32F;
      break;
    case mediapipe::ImageFormat::VEC32F2:
      type = CV_32FC2;
      break;
    case mediapipe::ImageFormat::VEC32F4:
      type = CV_32FC4;
      break;
    case mediapipe::ImageFormat::LAB8:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SBGRA:
      type = CV_8U;
      break;
    default:
      // Invalid or unknown; Default to uchar.
      type = CV_8U;
      break;
  }
  return type;
}
}  // namespace

namespace mediapipe {
namespace formats {

cv::Mat MatView(const ImageFrame* image) {
  const int dims = 2;
  const int sizes[] = {image->Height(), image->Width()};
  const int type =
      CV_MAKETYPE(GetMatType(image->Format()), image->NumberOfChannels());
  const size_t steps[] = {static_cast<size_t>(image->WidthStep()),
                          static_cast<size_t>(image->ByteDepth())};
  // Use ImageFrame to initialize in-place. ImageFrame still owns memory.
  return cv::Mat(dims, sizes, type, const_cast<uint8_t*>(image->PixelData()),
                 steps);
}

}  // namespace formats
}  // namespace mediapipe
