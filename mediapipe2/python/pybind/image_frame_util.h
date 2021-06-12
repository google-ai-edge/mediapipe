// Copyright 2020-2021 The MediaPipe Authors.
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
#include "mediapipe/python/pybind/util.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

template <typename T>
std::unique_ptr<ImageFrame> CreateImageFrame(
    mediapipe::ImageFormat::Format format,
    const py::array_t<T, py::array::c_style>& data, bool copy = true) {
  int rows = data.shape()[0];
  int cols = data.shape()[1];
  int width_step = ImageFrame::NumberOfChannelsForFormat(format) *
                   ImageFrame::ByteDepthForFormat(format) * cols;
  if (copy) {
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
  PyObject* data_pyobject = data.ptr();
  auto image_frame = absl::make_unique<ImageFrame>(
      format, /*width=*/cols, /*height=*/rows, width_step,
      static_cast<uint8*>(data.request().ptr),
      /*deleter=*/[data_pyobject](uint8*) { Py_XDECREF(data_pyobject); });
  Py_XINCREF(data_pyobject);
  return image_frame;
}

template <typename T>
py::array GenerateContiguousDataArrayHelper(const ImageFrame& image_frame,
                                            const py::object& py_object) {
  std::vector<int> shape{image_frame.Height(), image_frame.Width()};
  if (image_frame.NumberOfChannels() > 1) {
    shape.push_back(image_frame.NumberOfChannels());
  }
  py::array_t<T, py::array::c_style> contiguous_data;
  if (image_frame.IsContiguous()) {
    contiguous_data = py::array_t<T, py::array::c_style>(
        shape, reinterpret_cast<const T*>(image_frame.PixelData()), py_object);
  } else {
    auto contiguous_data_copy =
        absl::make_unique<T[]>(image_frame.Width() * image_frame.Height() *
                               image_frame.NumberOfChannels());
    image_frame.CopyToBuffer(contiguous_data_copy.get(),
                             image_frame.PixelDataSizeStoredContiguously());
    auto capsule = py::capsule(contiguous_data_copy.get(), [](void* data) {
      if (data) {
        delete[] reinterpret_cast<T*>(data);
      }
    });
    contiguous_data = py::array_t<T, py::array::c_style>(
        shape, contiguous_data_copy.release(), capsule);
  }

  // In both cases, the underlying data is not writable in Python.
  py::detail::array_proxy(contiguous_data.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  return contiguous_data;
}

inline py::array GenerateContiguousDataArray(const ImageFrame& image_frame,
                                             const py::object& py_object) {
  switch (image_frame.ChannelSize()) {
    case sizeof(uint8):
      return GenerateContiguousDataArrayHelper<uint8>(image_frame, py_object)
          .cast<py::array>();
    case sizeof(uint16):
      return GenerateContiguousDataArrayHelper<uint16>(image_frame, py_object)
          .cast<py::array>();
    case sizeof(float):
      return GenerateContiguousDataArrayHelper<float>(image_frame, py_object)
          .cast<py::array>();
      break;
    default:
      throw RaisePyError(PyExc_RuntimeError,
                         "Unsupported image frame channel size. Data is not "
                         "uint8, uint16, or float?");
  }
}

// Generates a contiguous data pyarray object on demand.
// This function only accepts an image frame object that already stores
// contiguous data. The output py::array points to the raw pixel data array of
// the image frame object directly.
inline py::array GenerateDataPyArrayOnDemand(const ImageFrame& image_frame,
                                             const py::object& py_object) {
  if (!image_frame.IsContiguous()) {
    throw RaisePyError(PyExc_RuntimeError,
                       "GenerateDataPyArrayOnDemand must take an ImageFrame "
                       "object that stores contiguous data.");
  }
  return GenerateContiguousDataArray(image_frame, py_object);
}

// Gets the cached contiguous data array from the "__contiguous_data" attribute.
// If the attribute doesn't exist, the function calls
// GenerateContiguousDataArray() to generate the contiguous data pyarray object,
// which realigns and copies the data from the original image frame object.
// Then, the data array object is cached in the "__contiguous_data" attribute.
// This function only accepts an image frame object that stores non-contiguous
// data.
inline py::array GetCachedContiguousDataAttr(const ImageFrame& image_frame,
                                             const py::object& py_object) {
  if (image_frame.IsContiguous()) {
    throw RaisePyError(PyExc_RuntimeError,
                       "GetCachedContiguousDataAttr must take an ImageFrame "
                       "object that stores non-contiguous data.");
  }
  py::object get_data_attr =
      py::getattr(py_object, "__contiguous_data", py::none());
  if (image_frame.IsEmpty()) {
    throw RaisePyError(PyExc_RuntimeError, "ImageFrame is unallocated.");
  }
  // If __contiguous_data attr doesn't store data yet, generates the contiguous
  // data array object and caches the result.
  if (get_data_attr.is_none()) {
    py_object.attr("__contiguous_data") =
        GenerateContiguousDataArray(image_frame, py_object);
  }
  return py_object.attr("__contiguous_data").cast<py::array>();
}

template <typename T>
py::object GetValue(const ImageFrame& image_frame, const std::vector<int>& pos,
                    const py::object& py_object) {
  py::array_t<T, py::array::c_style> output_array =
      image_frame.IsContiguous()
          ? GenerateDataPyArrayOnDemand(image_frame, py_object)
          : GetCachedContiguousDataAttr(image_frame, py_object);
  if (pos.size() == 2) {
    return py::cast(static_cast<T>(output_array.at(pos[0], pos[1])));
  } else if (pos.size() == 3) {
    return py::cast(static_cast<T>(output_array.at(pos[0], pos[1], pos[2])));
  }
  return py::none();
}

}  // namespace python
}  // namespace mediapipe

#endif  // MEDIAPIPE_PYTHON_PYBIND_IMAGE_FRAME_UTIL_H_
