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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "libyuv/convert_argb.h"
#include "libyuv/video_common.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/yuv_image.h"

namespace mediapipe {
namespace api2 {

namespace {

// Utility function to convert FourCC enum to string, for error messages.
std::string FourCCToString(libyuv::FourCC fourcc) {
  char buf[5];
  buf[0] = (fourcc >> 24) & 0xff;
  buf[1] = (fourcc >> 16) & 0xff;
  buf[2] = (fourcc >> 8) & 0xff;
  buf[3] = (fourcc)&0xff;
  buf[4] = 0;
  return std::string(buf);
}
}  // namespace

// Converts a `YUVImage` into an RGB `Image` using libyuv.
//
// The input `YUVImage` is expected to be in the NV12, NV21, YV12 or I420 (aka
// YV21) format (as per the `fourcc()` property). This covers the most commonly
// used YUV image formats used on mobile devices. Other formats are not
// supported and wil result in an `InvalidArgumentError`.
class YUVToImageCalculator : public Node {
 public:
  static constexpr Input<YUVImage> kInput{"YUV_IMAGE"};
  static constexpr Output<Image> kOutput{"IMAGE"};

  MEDIAPIPE_NODE_CONTRACT(kInput, kOutput);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& yuv_image = *kInput(cc);
    // Check that the format is supported.
    auto format = yuv_image.fourcc();
    if (format != libyuv::FOURCC_NV12 && format != libyuv::FOURCC_NV21 &&
        format != libyuv::FOURCC_YV12 && format != libyuv::FOURCC_I420) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported YUVImage format: %s. Only NV12, NV21, "
                          "YV12 and I420 (aka YV21) are supported.",
                          FourCCToString(format)));
    }
    // Build a transient ImageFrameSharedPtr with default alignment to host
    // conversion results.
    ImageFrameSharedPtr image_frame = std::make_shared<ImageFrame>(
        ImageFormat::SRGB, yuv_image.width(), yuv_image.height());
    // Perform actual conversion.
    switch (format) {
      case libyuv::FOURCC_NV12:
        // 8-bit Y plane followed by an interleaved 8-bit U/V plane with 2×2
        // subsampling.
        libyuv::NV12ToRAW(
            yuv_image.data(0), yuv_image.stride(0), yuv_image.data(1),
            yuv_image.stride(1), image_frame->MutablePixelData(),
            image_frame->WidthStep(), yuv_image.width(), yuv_image.height());
        break;
      case libyuv::FOURCC_NV21:
        // 8-bit Y plane followed by an interleaved 8-bit V/U plane with 2×2
        // subsampling.
        libyuv::NV21ToRAW(
            yuv_image.data(0), yuv_image.stride(0), yuv_image.data(1),
            yuv_image.stride(1), image_frame->MutablePixelData(),
            image_frame->WidthStep(), yuv_image.width(), yuv_image.height());
        break;
      case libyuv::FOURCC_I420:
        // Also known as YV21.
        // 8-bit Y plane followed by 8-bit 2×2 subsampled U and V planes.
        libyuv::I420ToRAW(
            yuv_image.data(0), yuv_image.stride(0), yuv_image.data(1),
            yuv_image.stride(1), yuv_image.data(2), yuv_image.stride(2),
            image_frame->MutablePixelData(), image_frame->WidthStep(),
            yuv_image.width(), yuv_image.height());
        break;
      case libyuv::FOURCC_YV12:
        // 8-bit Y plane followed by 8-bit 2×2 subsampled V and U planes.
        libyuv::I420ToRAW(
            yuv_image.data(0), yuv_image.stride(0), yuv_image.data(2),
            yuv_image.stride(2), yuv_image.data(1), yuv_image.stride(1),
            image_frame->MutablePixelData(), image_frame->WidthStep(),
            yuv_image.width(), yuv_image.height());
        break;
      default:
        // This should never happen (caught by checks above).
        return absl::InternalError("Unsupported YUVImage format.");
    }
    // Finally, build and send an Image object that takes ownership of the
    // transient ImageFrameSharedPtr object.
    kOutput(cc).Send(std::make_unique<Image>(std::move(image_frame)));
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(YUVToImageCalculator);

}  // namespace api2
}  // namespace mediapipe
