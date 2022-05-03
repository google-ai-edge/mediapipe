// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/image.h"

#include <memory>

#include "mediapipe/python/pybind/image_frame_util.h"
#include "mediapipe/python/pybind/util.h"
#include "pybind11/stl.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

void ImageSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule("image", "MediaPipe image module");

  py::options options;
  options.disable_function_signatures();

  // Image
  py::class_<Image> image(
      m, "Image",
      R"doc(A container for storing an image or a video frame, in one of several formats.

  Formats supported by Image are listed in the ImageFormat enum.
  Pixels are encoded row-major in an interleaved fashion. Image supports
  uint8, uint16, and float as its data types.

  Image can be created by copying the data from a numpy ndarray that stores
  the pixel data continuously. An Image may realign the input data on its
  default alignment boundary during creation. The data in an Image will
  become immutable after creation.

  Creation examples:
    import cv2
    cv_mat = cv2.imread(input_file)[:, :, ::-1]
    rgb_frame = mp.Image(format=ImageFormat.SRGB, data=cv_mat)
    gray_frame = mp.Image(
        format=ImageFormat.GRAY, data=cv2.cvtColor(cv_mat, cv2.COLOR_RGB2GRAY))

    from PIL import Image
    pil_img = Image.new('RGB', (60, 30), color = 'red')
    image = mp.Image(
        format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

  The pixel data in an Image can be retrieved as a numpy ndarray by calling
  `Image.numpy_view()`. The returned numpy ndarray is a reference to the
  internal data and itself is unwritable. If the callers want to modify the
  numpy ndarray, it's required to obtain a copy of it.

  Pixel data retrieval examples:
    for channel in range(num_channel):
      for col in range(width):
        for row in range(height):
          print(image[row, col, channel])

    output_ndarray = image.numpy_view()
    print(output_ndarray[0, 0, 0])
    copied_ndarray = np.copy(output_ndarray)
    copied_ndarray[0,0,0] = 0
  )doc",
      py::dynamic_attr());

  image
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
            return Image(std::shared_ptr<ImageFrame>(
                CreateImageFrame<uint8>(format, data)));
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
            return Image(std::shared_ptr<ImageFrame>(
                CreateImageFrame<uint16>(format, data)));
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
            return Image(std::shared_ptr<ImageFrame>(
                CreateImageFrame<float>(format, data)));
          }),
          R"doc(For float data type, valid ImageFormat are VEC32F1 and VEC32F2.)doc",
          py::arg("image_format"), py::arg("data").noconvert());

  image.def(
      "numpy_view",
      [](Image& self) {
        py::object py_object =
            py::cast(self, py::return_value_policy::reference);
        // If the image data is contiguous, generates the data pyarray object
        // on demand because 1) making a pyarray by referring to the existing
        // image pixel data is relatively cheap and 2) caching the pyarray
        // object in an attribute of the image is problematic: the image object
        // and the data pyarray object refer to each other, which causes gc
        // fails to free the pyarray after use.
        // For the non-contiguous cases, gets a cached data pyarray object from
        // the image pyobject attribute. This optimization is to avoid the
        // expensive data realignment and copy operations happening more than
        // once.
        return self.GetImageFrameSharedPtr()->IsContiguous()
                   ? GenerateDataPyArrayOnDemand(*self.GetImageFrameSharedPtr(),
                                                 py_object)
                   : GetCachedContiguousDataAttr(*self.GetImageFrameSharedPtr(),
                                                 py_object);
      },
      R"doc(Return the image pixel data as an unwritable numpy ndarray.

  Realign the pixel data to be stored contiguously and return a reference to the
  unwritable numpy ndarray. If the callers want to modify the numpy array data,
  it's required to obtain a copy of the ndarray.

  Returns:
    An unwritable numpy ndarray.

  Examples:
    output_ndarray = image.numpy_view()
    copied_ndarray = np.copy(output_ndarray)
    copied_ndarray[0,0,0] = 0
)doc");

  image.def(
      "__getitem__",
      [](Image& self, const std::vector<int>& pos) {
        if (pos.size() != 3 && !(pos.size() == 2 && self.channels() == 1)) {
          throw RaisePyError(
              PyExc_IndexError,
              absl::StrCat("Invalid index dimension: ", pos.size()).c_str());
        }
        py::object py_object =
            py::cast(self, py::return_value_policy::reference);
        switch (self.GetImageFrameSharedPtr()->ByteDepth()) {
          case 1:
            return GetValue<uint8>(*self.GetImageFrameSharedPtr(), pos,
                                   py_object);
          case 2:
            return GetValue<uint16>(*self.GetImageFrameSharedPtr(), pos,
                                    py_object);
          case 4:
            return GetValue<float>(*self.GetImageFrameSharedPtr(), pos,
                                   py_object);
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
          print(image[row, col, channel])
)doc");

  image
      .def("uses_gpu", &Image::UsesGpu,
           R"doc(Return True if data is currently on the GPU.)doc")
      .def(
          "is_contiguous",
          [](Image& self) {
            return self.GetImageFrameSharedPtr()->IsContiguous();
          },
          R"doc(Return True if the pixel data is stored contiguously (without any alignment padding areas).)doc")
      .def(
          "is_empty",
          [](Image& self) { return self.GetImageFrameSharedPtr()->IsEmpty(); },
          R"doc(Return True if the pixel data is unallocated.)doc")
      .def(
          "is_aligned",
          [](Image& self, uint32 alignment_boundary) {
            return self.GetImageFrameSharedPtr()->IsAligned(alignment_boundary);
          },
          R"doc(Return True if each row of the data is aligned to alignment boundary, which must be 1 or a power of 2.

  Args:
    alignment_boundary: An integer.

  Returns:
    A boolean.

  Examples:
    image.is_aligned(16)
)doc");

  image.def_property_readonly("width", &Image::width)
      .def_property_readonly("height", &Image::height)
      .def_property_readonly("channels", &Image::channels)
      .def_property_readonly("step", &Image::step)
      .def_property_readonly("image_format", &Image::image_format);
}

}  // namespace python
}  // namespace mediapipe
