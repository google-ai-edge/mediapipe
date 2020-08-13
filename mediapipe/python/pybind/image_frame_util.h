// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_PYTHON_PYBIND_IMAGE_FRAME_UTIL_H_
#define MEDIAPIPE_PYTHON_PYBIND_IMAGE_FRAME_UTIL_H_

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

// TODO: Implement the reference mode of image frame creation, which
// takes a reference to the external data rather than copying it over.
// A possible solution is to have a custom PixelDataDeleter:
// The refcount of the numpy array will be increased when the image frame is
// created by taking a reference to the external numpy array data. Then, the
// custom PixelDataDeleter will decrease the refcount when the image frame gets
// destroyed and let Python GC does its job.
template <typename T>
std::unique_ptr<ImageFrame> CreateImageFrame(
    mediapipe::ImageFormat::Format format,
    const py::array_t<T, py::array::c_style>& data) {
  int rows = data.shape()[0];
  int cols = data.shape()[1];
  int width_step = ImageFrame::NumberOfChannelsForFormat(format) *
                   ImageFrame::ByteDepthForFormat(format) * cols;
  auto image_frame = absl::make_unique<ImageFrame>(
      format, /*width=*/cols, /*height=*/rows, width_step,
      static_cast<uint8*>(data.request().ptr),
      ImageFrame::PixelDataDeleter::kNone);
  auto image_frame_copy = absl::make_unique<ImageFrame>();
  // Set alignment_boundary to kGlDefaultAlignmentBoundary so that both
  // GPU and CPU can process it.
  image_frame_copy->CopyFrom(*image_frame,
                             ImageFrame::kGlDefaultAlignmentBoundary);
  return image_frame_copy;
}

}  // namespace python
}  // namespace mediapipe

#endif  // MEDIAPIPE_PYTHON_PYBIND_IMAGE_FRAME_UTIL_H_
