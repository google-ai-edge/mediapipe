// Copyright 2026 The MediaPipe Authors.
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

#ifndef __EMSCRIPTEN__
#error "This is web-only code, but was built for a non-web target platform."
#endif  // __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/em_macros.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter/proto/stroke.pb.h"
#include "mediapipe/tasks/web/vision/interactive_segmenter/mask_util.h"
#include "mediapipe/web/wasm_error_helper.h"

namespace {

using ::mediapipe::web::InvokeErrorListener;

using ::mediapipe::Image;
using ::mediapipe::ImageFormat;
using ::mediapipe::ImageFrame;
using ::mediapipe::tasks::vision::interactive_segmenter::InteractiveSegmenter;
using ::mediapipe::tasks::vision::interactive_segmenter::
    InteractiveSegmenterOptions;
using ::mediapipe::tasks::web::vision::CopyMask;

}  // namespace

extern "C" {

// Creates the native C++ InteractiveSegmenter engine instance.
// Options are serialized in JS as BaseOptions proto and passed via Wasm heap.
EMSCRIPTEN_KEEPALIVE intptr_t
interactive_segmenter_create(uint8_t* base_options_ptr, int base_options_size) {
  mediapipe::tasks::core::proto::BaseOptions base_options_proto;
  if (!base_options_proto.ParseFromArray(base_options_ptr, base_options_size)) {
    InvokeErrorListener(
        absl::InvalidArgumentError("Failed to parse BaseOptions proto."));
    return 0;
  }

  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options = mediapipe::tasks::core::ConvertProtoToBaseOptions(
      std::move(base_options_proto));

  absl::StatusOr<std::unique_ptr<InteractiveSegmenter>> segmenter_or =
      InteractiveSegmenter::Create(std::move(options));
  if (!segmenter_or.ok()) {
    InvokeErrorListener(segmenter_or.status());
    return 0;
  }
  return reinterpret_cast<intptr_t>(segmenter_or.value().release());
}

// Sets the input image for segmentation, executing the encoder model.
// Copies pixel data from Wasm heap into a CPU-bound mediapipe::Image container.
EMSCRIPTEN_KEEPALIVE bool interactive_segmenter_set_image(
    intptr_t segmenter_handle, uint8_t* pixel_data_ptr, int width, int height,
    int num_channels) {
  InteractiveSegmenter* segmenter =
      reinterpret_cast<InteractiveSegmenter*>(segmenter_handle);
  if (!segmenter || !pixel_data_ptr) {
    InvokeErrorListener(absl::InvalidArgumentError(
        "Invalid segmenter handle or pixel data pointer."));
    return false;
  }

  ImageFormat::Format format = ImageFormat::SRGBA;
  if (num_channels == 3) {
    format = ImageFormat::SRGB;
  } else if (num_channels == 4) {
    format = ImageFormat::SRGBA;
  } else if (num_channels == 1) {
    format = ImageFormat::GRAY8;
  } else {
    InvokeErrorListener(absl::InvalidArgumentError(
        "Unsupported number of channels for input image."));
    return false;
  }

  auto image_frame = std::make_unique<ImageFrame>(format, width, height);
  for (int y = 0; y < height; ++y) {
    std::memcpy(image_frame->MutablePixelData() + y * image_frame->WidthStep(),
                pixel_data_ptr + y * width * num_channels,
                width * num_channels);
  }

  Image image(std::move(image_frame));
  absl::Status status = segmenter->SetImage(image);
  if (!status.ok()) {
    InvokeErrorListener(status);
    return false;
  }
  return true;
}

// Executes the lightweight decoder model with user strokes.
// Extracts output dimensions and dynamically allocates Wasm heap memory for the
// resulting mask.
EMSCRIPTEN_KEEPALIVE intptr_t interactive_segmenter_segment(
    intptr_t segmenter_handle, uint8_t* strokes_proto_ptr,
    int strokes_proto_size, int* out_width, int* out_height, int* out_size) {
  InteractiveSegmenter* segmenter =
      reinterpret_cast<InteractiveSegmenter*>(segmenter_handle);
  if (!segmenter || !strokes_proto_ptr) {
    InvokeErrorListener(absl::InvalidArgumentError(
        "Invalid segmenter handle or strokes pointer."));
    return 0;
  }

  mediapipe::tasks::vision::interactive_segmenter::proto::Strokes strokes;
  if (!strokes.ParseFromArray(strokes_proto_ptr, strokes_proto_size)) {
    InvokeErrorListener(
        absl::InvalidArgumentError("Failed to parse Strokes proto."));
    return 0;
  }

  absl::StatusOr<Image> mask_or = segmenter->Segment(strokes);
  if (!mask_or.ok()) {
    InvokeErrorListener(mask_or.status());
    return 0;
  }

  std::shared_ptr<const ImageFrame> frame =
      mask_or.value().GetImageFrameSharedPtr();
  if (!frame) {
    InvokeErrorListener(
        absl::InternalError("Failed to extract ImageFrame from mask Image."));
    return 0;
  }

  int width = frame->Width();
  int height = frame->Height();
  int channels = frame->NumberOfChannels();
  int channel_size = frame->ChannelSize();
  int width_step = frame->WidthStep();
  const uint8_t* src_data = frame->PixelData();

  *out_width = width;
  *out_height = height;
  *out_size = width * height * channel_size;

  uint8_t* heap_buf = reinterpret_cast<uint8_t*>(std::malloc(*out_size));
  if (!heap_buf) {
    InvokeErrorListener(absl::InternalError(
        "Failed to allocate Wasm heap buffer for output mask."));
    return 0;
  }

  absl::Status status = CopyMask(src_data, width, height, channels,
                                 channel_size, width_step, heap_buf);
  if (!status.ok()) {
    InvokeErrorListener(status);
    std::free(heap_buf);
    return 0;
  }
  return reinterpret_cast<intptr_t>(heap_buf);
}

// Safely deletes the native C++ engine instance.
EMSCRIPTEN_KEEPALIVE void interactive_segmenter_close(
    intptr_t segmenter_handle) {
  InteractiveSegmenter* segmenter =
      reinterpret_cast<InteractiveSegmenter*>(segmenter_handle);
  delete segmenter;
}

}  // extern "C"
