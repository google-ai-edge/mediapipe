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

#include "mediapipe/python/pybind/image_frame_util.h"
#include "mediapipe/python/pybind/util.h"
#include "pybind11/stl.h"

namespace mediapipe {
namespace python {
namespace {

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

py::array GenerateContiguousDataArray(const ImageFrame& image_frame,
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
py::array GenerateDataPyArrayOnDemand(const ImageFrame& image_frame,
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
py::array GetCachedContiguousDataAttr(const ImageFrame& image_frame,
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

}  // namespace

namespace py = pybind11;

void ImageFrameSubmodule(pybind11::module* module) {
  py::module m =
      module->def_submodule("image_frame", "MediaPipe image frame module");

  py::options options;
  options.disable_function_signatures();

  // ImageFormat
  py::enum_<mediapipe::ImageFormat::Format> image_format(
      m, "ImageFormat",
      R"doc(An enum describing supported raw image formats.

  SRGB: sRGB, interleaved: one byte for R, then one byte for G, then one byte
    for B for each pixel.

  SRGBA: sRGBA, interleaved: one byte for R, one byte for G, one byte for B, one
    byte for alpha or unused.

  SBGRA: sBGRA, interleaved: one byte for B, one byte for G, one byte for R, one
    byte for alpha or unused.

  GRAY8: Grayscale, one byte per pixel.

  GRAY16: Grayscale, one uint16 per pixel.

  SRGB48: sRGB, interleaved, each component is a uint16.

  SRGBA64: sRGBA, interleaved, each component is a uint16.

  VEC32F1: One float per pixel.

  VEC32F2: Two floats per pixel.
)doc");

  image_format.value("SRGB", mediapipe::ImageFormat::SRGB)
      .value("SRGBA", mediapipe::ImageFormat::SRGBA)
      .value("SBGRA", mediapipe::ImageFormat::SBGRA)
      .value("GRAY8", mediapipe::ImageFormat::GRAY8)
      .value("GRAY16", mediapipe::ImageFormat::GRAY16)
      .value("SRGB48", mediapipe::ImageFormat::SRGB48)
      .value("SRGBA64", mediapipe::ImageFormat::SRGBA64)
      .value("VEC32F1", mediapipe::ImageFormat::VEC32F1)
      .value("VEC32F2", mediapipe::ImageFormat::VEC32F2)
      .export_values();

  // ImageFrame
  py::class_<ImageFrame> image_frame(
      m, "ImageFrame",
      R"doc(A container for storing an image or a video frame, in one of several formats.

  Formats supported by ImageFrame are listed in the ImageFormat enum.
  Pixels are encoded row-major in an interleaved fashion. ImageFrame supports
  uint8, uint16, and float as its data types.

  ImageFrame can be created by copying the data from a numpy ndarray that stores
  the pixel data continuously. An ImageFrame may realign the input data on its
  default alignment boundary during creation. The data in an ImageFrame will
  become immutable after creation.

  Creation examples:
    import cv2
    cv_mat = cv2.imread(input_file)[:, :, ::-1]
    rgb_frame = mp.ImageFrame(format=ImageFormat.SRGB, data=cv_mat)
    gray_frame = mp.ImageFrame(
        format=ImageFormat.GRAY, data=cv2.cvtColor(cv_mat, cv2.COLOR_RGB2GRAY))

    from PIL import Image
    pil_img = Image.new('RGB', (60, 30), color = 'red')
    image_frame = mp.ImageFrame(
        format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

  The pixel data in an ImageFrame can be retrieved as a numpy ndarray by calling
  `ImageFrame.numpy_view()`. The returned numpy ndarray is a reference to the
  internal data and itself is unwritable. If the callers want to modify the
  numpy ndarray, it's required to obtain a copy of it.

  Pixel data retrieval examples:
    for channel in range(num_channel):
      for col in range(width):
        for row in range(height):
          print(image_frame[row, col, channel])

    output_ndarray = image_frame.numpy_view()
    print(output_ndarray[0, 0, 0])
    copied_ndarray = np.copy(output_ndarray)
    copied_ndarray[0,0,0] = 0
  )doc",
      py::dynamic_attr());

  image_frame
      .def(
          py::init([](mediapipe::ImageFormat::Format format,
                      const py::array_t<uint8, py::array::c_style>& data) {
            if (format != mediapipe::ImageFormat::GRAY8 &&
                format != mediapipe::ImageFormat::SRGB &&
                format != mediapipe::ImageFormat::SRGBA) {
              throw RaisePyError(PyExc_RuntimeError,
                                 "uint8 image data should be one of the GRAY8, "
                                 "SRGB, and SRGBA MediaPipe image formats.");
            }
            return CreateImageFrame<uint8>(format, data);
          }),
          R"doc(For uint8 data type, valid ImageFormat are GRAY8, SGRB, and SRGBA.)doc",
          py::arg("image_format"), py::arg("data").noconvert())
      .def(
          py::init([](mediapipe::ImageFormat::Format format,
                      const py::array_t<uint16, py::array::c_style>& data) {
            if (format != mediapipe::ImageFormat::GRAY16 &&
                format != mediapipe::ImageFormat::SRGB48 &&
                format != mediapipe::ImageFormat::SRGBA64) {
              throw RaisePyError(
                  PyExc_RuntimeError,
                  "uint16 image data should be one of the GRAY16, "
                  "SRGB48, and SRGBA64 MediaPipe image formats.");
            }
            return CreateImageFrame<uint16>(format, data);
          }),
          R"doc(For uint16 data type, valid ImageFormat are GRAY16, SRGB48, and SRGBA64.)doc",
          py::arg("image_format"), py::arg("data").noconvert())
      .def(
          py::init([](mediapipe::ImageFormat::Format format,
                      const py::array_t<float, py::array::c_style>& data) {
            if (format != mediapipe::ImageFormat::VEC32F1 &&
                format != mediapipe::ImageFormat::VEC32F2) {
              throw RaisePyError(
                  PyExc_RuntimeError,
                  "float image data should be either VEC32F1 or VEC32F2 "
                  "MediaPipe image formats.");
            }
            return CreateImageFrame<float>(format, data);
          }),
          R"doc(For float data type, valid ImageFormat are VEC32F1 and VEC32F2.)doc",
          py::arg("image_format"), py::arg("data").noconvert());

  image_frame.def(
      "numpy_view",
      [](ImageFrame& self) {
        py::object py_object =
            py::cast(self, py::return_value_policy::reference);
        // If the image frame data is contiguous, generates the data pyarray
        // object on demand because 1) making a pyarray by referring to the
        // existing image frame pixel data is relatively cheap and 2) caching
        // the pyarray object in an attribute of the image frame is problematic:
        // the image frame object and the data pyarray object refer to each
        // other, which causes gc fails to free the pyarray after use.
        // For the non-contiguous cases, gets a cached data pyarray object from
        // the image frame pyobject attribute. This optimization is to avoid the
        // expensive data realignment and copy operations happening more than
        // once.
        return self.IsContiguous()
                   ? GenerateDataPyArrayOnDemand(self, py_object)
                   : GetCachedContiguousDataAttr(self, py_object);
      },
      R"doc(Return the image frame pixel data as an unwritable numpy ndarray.

  Realign the pixel data to be stored contiguously and return a reference to the
  unwritable numpy ndarray. If the callers want to modify the numpy array data,
  it's required to obtain a copy of the ndarray.

  Returns:
    An unwritable numpy ndarray.

  Examples:
    output_ndarray = image_frame.numpy_view()
    copied_ndarray = np.copy(output_ndarray)
    copied_ndarray[0,0,0] = 0
)doc");

  image_frame.def(
      "__getitem__",
      [](ImageFrame& self, const std::vector<int>& pos) {
        if (pos.size() != 3 &&
            !(pos.size() == 2 && self.NumberOfChannels() == 1)) {
          throw RaisePyError(
              PyExc_IndexError,
              absl::StrCat("Invalid index dimension: ", pos.size()).c_str());
        }
        py::object py_object =
            py::cast(self, py::return_value_policy::reference);
        switch (self.ByteDepth()) {
          case 1:
            return GetValue<uint8>(self, pos, py_object);
          case 2:
            return GetValue<uint16>(self, pos, py_object);
          case 4:
            return GetValue<float>(self, pos, py_object);
          default:
            return py::object();
        }
      },
      R"doc(Use the indexer operators to access pixel data.

  Raises:
    IndexError: If the index is invalid or out of bounds.

  Examples:
    for channel in range(num_channel):
      for col in range(width):
        for row in range(height):
          print(image_frame[row, col, channel])

)doc");

  image_frame
      .def(
          "is_contiguous", &ImageFrame::IsContiguous,
          R"doc(Return True if the pixel data is stored contiguously (without any alignment padding areas).)doc")
      .def("is_empty", &ImageFrame::IsEmpty,
           R"doc(Return True if the pixel data is unallocated.)doc")
      .def(
          "is_aligned", &ImageFrame::IsAligned,
          R"doc(Return True if each row of the data is aligned to alignment boundary, which must be 1 or a power of 2.

  Args:
    alignment_boundary: An integer.

  Returns:
    A boolean.

  Examples:
    image_frame.is_aligned(16)
)doc");

  image_frame.def_property_readonly("width", &ImageFrame::Width)
      .def_property_readonly("height", &ImageFrame::Height)
      .def_property_readonly("channels", &ImageFrame::NumberOfChannels)
      .def_property_readonly("byte_depth", &ImageFrame::ByteDepth)
      .def_property_readonly("image_format", &ImageFrame::Format);
}

}  // namespace python
}  // namespace mediapipe
