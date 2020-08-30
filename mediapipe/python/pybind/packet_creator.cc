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

#include "mediapipe/python/pybind/packet_creator.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/python/pybind/image_frame_util.h"
#include "mediapipe/python/pybind/util.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mediapipe {
namespace python {
namespace {

Packet CreateImageFramePacket(mediapipe::ImageFormat::Format format,
                              const py::array& data, bool copy) {
  if (format == mediapipe::ImageFormat::SRGB ||
      format == mediapipe::ImageFormat::SRGBA ||
      format == mediapipe::ImageFormat::GRAY8) {
    return Adopt(CreateImageFrame<uint8>(format, data, copy).release());
  } else if (format == mediapipe::ImageFormat::GRAY16 ||
             format == mediapipe::ImageFormat::SRGB48 ||
             format == mediapipe::ImageFormat::SRGBA64) {
    return Adopt(CreateImageFrame<uint16>(format, data, copy).release());
  } else if (format == mediapipe::ImageFormat::VEC32F1 ||
             format == mediapipe::ImageFormat::VEC32F2) {
    return Adopt(CreateImageFrame<float>(format, data, copy).release());
  }
  throw RaisePyError(PyExc_RuntimeError,
                     absl::StrCat("Unsupported ImageFormat: ", format).c_str());
  return Packet();
}

}  // namespace

namespace py = pybind11;

void PublicPacketCreators(pybind11::module* m) {
  m->def(
      "create_string",
      [](const std::string& data) { return MakePacket<std::string>(data); },
      R"doc(Create a MediaPipe std::string Packet from a str.

  Args:
    data: A str.

  Returns:
    A MediaPipe std::string Packet.

  Raises:
    TypeError: If the input is not a str.

  Examples:
    packet = mp.packet_creator.create_string('abc')
    data = mp.packet_getter.get_string(packet)
)doc",
      py::return_value_policy::move);

  m->def(
      "create_string",
      [](const py::bytes& data) { return MakePacket<std::string>(data); },
      R"doc(Create a MediaPipe std::string Packet from a bytes object.

  Args:
    data: A bytes object.

  Returns:
    A MediaPipe std::string Packet.

  Raises:
    TypeError: If the input is not a bytes object.

  Examples:
    packet = mp.packet_creator.create_string(b'\xd0\xd0\xd0')
    data = mp.packet_getter.get_bytes(packet)
)doc",
      py::return_value_policy::move);

  m->def(
      "create_bool", [](bool data) { return MakePacket<bool>(data); },
      R"doc(Create a MediaPipe bool Packet from a boolean object.

  Args:
    data: A boolean object.

  Returns:
    A MediaPipe bool Packet.

  Raises:
    TypeError: If the input is not a boolean object.

  Examples:
    packet = mp.packet_creator.create_bool(True)
    data = mp.packet_getter.get_bool(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, INT_MIN, INT_MAX);
        return MakePacket<int>(data);
      },
      R"doc(Create a MediaPipe int Packet from an integer.

  Args:
    data: An integer or a np.intc.

  Returns:
    A MediaPipe int Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is not an integer.

  Examples:
    packet = mp.packet_creator.create_int(0)
    data = mp.packet_getter.get_int(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int8",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, INT8_MIN, INT8_MAX);
        return MakePacket<int8>(data);
      },
      R"doc(Create a MediaPipe int8 Packet from an integer.

  Args:
    data: An integer or a np.int8.

  Returns:
    A MediaPipe int8 Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is neither an integer nor a np.int8.

  Examples:
    packet = mp.packet_creator.create_int8(2**7 - 1)
    data = mp.packet_getter.get_int(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int16",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, INT16_MIN, INT16_MAX);
        return MakePacket<int16>(data);
      },
      R"doc(Create a MediaPipe int16 Packet from an integer.

  Args:
    data: An integer or a np.int16.

  Returns:
    A MediaPipe int16 Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is neither an integer nor a np.int16.

  Examples:
    packet = mp.packet_creator.create_int16(2**15 - 1)
    data = mp.packet_getter.get_int(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int32",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, INT32_MIN, INT32_MAX);
        return MakePacket<int32>(data);
      },
      R"doc(Create a MediaPipe int32 Packet from an integer.

  Args:
    data: An integer or a np.int32.

  Returns:
    A MediaPipe int32 Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is neither an integer nor a np.int32.

  Examples:
    packet = mp.packet_creator.create_int32(2**31 - 1)
    data = mp.packet_getter.get_int(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int64", [](int64 data) { return MakePacket<int64>(data); },
      R"doc(Create a MediaPipe int64 Packet from an integer.

  Args:
    data: An integer or a np.int64.

  Returns:
    A MediaPipe int64 Packet.

  Raises:
    TypeError: If the input is neither an integer nor a np.int64.

  Examples:
    packet = mp.packet_creator.create_int64(2**63 - 1)
    data = mp.packet_getter.get_int(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_uint8",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, 0, UINT8_MAX);
        return MakePacket<uint8>(data);
      },
      R"doc(Create a MediaPipe uint8 Packet from an integer.

  Args:
    data: An integer or a np.uint8.

  Returns:
    A MediaPipe uint8 Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is neither an integer nor a np.uint8.

  Examples:
    packet = mp.packet_creator.create_uint8(2**8 - 1)
    data = mp.packet_getter.get_uint(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_uint16",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, 0, UINT16_MAX);
        return MakePacket<uint16>(data);
      },
      R"doc(Create a MediaPipe uint16 Packet from an integer.

  Args:
    data: An integer or a np.uint16.

  Returns:
    A MediaPipe uint16 Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is neither an integer nor a np.uint16.

  Examples:
    packet = mp.packet_creator.create_uint16(2**16 - 1)
    data = mp.packet_getter.get_uint(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_uint32",
      [](int64 data) {
        RaisePyErrorIfOverflow(data, 0, UINT32_MAX);
        return MakePacket<uint32>(data);
      },
      R"doc(Create a MediaPipe uint32 Packet from an integer.

  Args:
    data: An integer or a np.uint32.

  Returns:
    A MediaPipe uint32 Packet.

  Raises:
    OverflowError: If the input integer overflows.
    TypeError: If the input is neither an integer nor a np.uint32.

  Examples:
    packet = mp.packet_creator.create_uint32(2**32 - 1)
    data = mp.packet_getter.get_uint(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_uint64", [](uint64 data) { return MakePacket<uint64>(data); },
      R"doc(Create a MediaPipe uint64 Packet from an integer.

  Args:
    data: An integer or a np.uint64.

  Returns:
    A MediaPipe uint64 Packet.

  Raises:
    TypeError: If the input is neither an integer nor a np.uint64.

  Examples:
    packet = mp.packet_creator.create_uint64(2**64 - 1)
    data = mp.packet_getter.get_uint(packet)
)doc",
      // py::arg().noconvert() won't allow this to accept np.uint64 data type.
      py::arg(), py::return_value_policy::move);

  m->def(
      "create_float", [](float data) { return MakePacket<float>(data); },
      R"doc(Create a MediaPipe float Packet from a float.

  Args:
    data: A float or a np.float.

  Returns:
    A MediaPipe float Packet.

  Raises:
    TypeError: If the input is neither a float nor a np.float.

  Examples:
    packet = mp.packet_creator.create_float(0.1)
    data = mp.packet_getter.get_float(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_double", [](double data) { return MakePacket<double>(data); },
      R"doc(Create a MediaPipe double Packet from a float.

  Args:
    data: A float or a np.double.

  Returns:
    A MediaPipe double Packet.

  Raises:
    TypeError: If the input is neither a float nore a np.double.

  Examples:
    packet = mp.packet_creator.create_double(0.1)
    data = mp.packet_getter.get_float(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int_array",
      [](const std::vector<int>& data) {
        int* ints = new int[data.size()];
        std::copy(data.begin(), data.end(), ints);
        return Adopt(reinterpret_cast<int(*)[]>(ints));
      },
      R"doc(Create a MediaPipe int array Packet from a list of integers.

  Args:
    data: A list of integers.

  Returns:
    A MediaPipe int array Packet.

  Raises:
    TypeError: If the input is not a list of integers.

  Examples:
    packet = mp.packet_creator.create_int_array([1, 2, 3])
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_float_array",
      [](const std::vector<float>& data) {
        float* floats = new float[data.size()];
        std::copy(data.begin(), data.end(), floats);
        return Adopt(reinterpret_cast<float(*)[]>(floats));
      },
      R"doc(Create a MediaPipe float array Packet from a list of floats.

  Args:
    data: A list of floats.

  Returns:
    A MediaPipe float array Packet.

  Raises:
    TypeError: If the input is not a list of floats.

  Examples:
    packet = mp.packet_creator.create_float_array([0.1, 0.2, 0.3])
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_int_vector",
      [](const std::vector<int>& data) {
        return MakePacket<std::vector<int>>(data);
      },
      R"doc(Create a MediaPipe int vector Packet from a list of integers.

  Args:
    data: A list of integers.

  Returns:
    A MediaPipe int vector Packet.

  Raises:
    TypeError: If the input is not a list of integers.

  Examples:
    packet = mp.packet_creator.create_int_vector([1, 2, 3])
    data = mp.packet_getter.get_int_vector(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_float_vector",
      [](const std::vector<float>& data) {
        return MakePacket<std::vector<float>>(data);
      },
      R"doc(Create a MediaPipe float vector Packet from a list of floats.

  Args:
    data: A list of floats

  Returns:
    A MediaPipe float vector Packet.

  Raises:
    TypeError: If the input is not a list of floats.

  Examples:
    packet = mp.packet_creator.create_float_vector([0.1, 0.2, 0.3])
    data = mp.packet_getter.get_float_list(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_string_vector",
      [](const std::vector<std::string>& data) {
        return MakePacket<std::vector<std::string>>(data);
      },
      R"doc(Create a MediaPipe std::string vector Packet from a list of str.

  Args:
    data: A list of str.

  Returns:
    A MediaPipe std::string vector Packet.

  Raises:
    TypeError: If the input is not a list of str.

  Examples:
    packet = mp.packet_creator.create_string_vector(['a', 'b', 'c'])
    data = mp.packet_getter.get_str_list(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_packet_vector",
      [](const std::vector<Packet>& data) {
        return MakePacket<std::vector<Packet>>(data);
      },
      R"doc(Create a MediaPipe Packet holds a vector of packets.

  Args:
    data: A list of packets.

  Returns:
    A MediaPipe Packet holds a vector of packets.

  Raises:
    TypeError: If the input is not a list of packets.

  Examples:
    packet = mp.packet_creator.create_packet_vector([
        mp.packet_creator.create_float(0.1),
        mp.packet_creator.create_int(1),
        mp.packet_creator.create_string('1')
    ])
   data = mp.packet_getter.get_packet_vector(packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_string_to_packet_map",
      [](const std::map<std::string, Packet>& data) {
        return MakePacket<std::map<std::string, Packet>>(data);
      },
      R"doc(Create a MediaPipe std::string to packet map Packet from a dictionary.

  Args:
    data: A dictionary that has (str, Packet) pairs.

  Returns:
    A MediaPipe Packet holds std::map<std::string, Packet>.

  Raises:
    TypeError: If the input is not a dictionary from str to packet.

  Examples:
    dict_packet = mp.packet_creator.create_string_to_packet_map({
        'float': mp.packet_creator.create_float(0.1),
        'int': mp.packet_creator.create_int(1),
        'std::string': mp.packet_creator.create_string('1')
    data = mp.packet_getter.get_str_to_packet_dict(dict_packet)
)doc",
      py::arg().noconvert(), py::return_value_policy::move);

  m->def(
      "create_matrix",
      // Eigen Map class
      // (https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html) is the
      // way to reuse the external memory as an Eigen type. However, when
      // creating an Eigen::MatrixXf from an Eigen Map object, the data copy
      // still happens. We can  make a packet of an Eigen Map type for reusing
      // external memory. However,the packet data type is no longer
      // Eigen::MatrixXf.
      // TODO: Should take "const Eigen::Ref<const Eigen::MatrixXf>&"
      // as the input argument. Investigate why bazel non-optimized mode
      // triggers a memory allocation bug in Eigen::internal::aligned_free().
      [](const Eigen::MatrixXf& matrix) {
        // MakePacket copies the data.
        return MakePacket<Matrix>(matrix);
      },
      R"doc(Create a MediaPipe Matrix Packet from a 2d numpy float ndarray.

  The method copies data from the input MatrixXf and the returned packet owns
  a MatrixXf object.

  Args:
    matrix: A 2d numpy float ndarray.

  Returns:
    A MediaPipe Matrix Packet.

  Raises:
    TypeError: If the input is not a 2d numpy float ndarray.

  Examples:
    packet = mp.packet_creator.create_matrix(
        np.array([[.1, .2, .3], [.4, .5, .6]])
    matrix = mp.packet_getter.get_matrix(packet)
)doc",
      py::return_value_policy::move);
}

void InternalPacketCreators(pybind11::module* m) {
  m->def("_create_image_frame_from_pixel_data", &CreateImageFramePacket,
         py::arg("format"), py::arg("data").noconvert(), py::arg("copy"),
         py::return_value_policy::move);

  m->def(
      "_create_image_frame_from_image_frame",
      [](ImageFrame& image_frame) {
        auto image_frame_copy = absl::make_unique<ImageFrame>();
        // Set alignment_boundary to kGlDefaultAlignmentBoundary so that
        // both GPU and CPU can process it.
        image_frame_copy->CopyFrom(image_frame,
                                   ImageFrame::kGlDefaultAlignmentBoundary);
        return Adopt(image_frame_copy.release());
      },
      py::arg("image_frame").noconvert(), py::return_value_policy::move);

  m->def(
      "_create_proto",
      [](const std::string& type_name, const py::bytes& serialized_proto) {
        using packet_internal::HolderBase;
        mediapipe::StatusOr<std::unique_ptr<HolderBase>> maybe_holder =
            packet_internal::MessageHolderRegistry::CreateByName(type_name);
        if (!maybe_holder.ok()) {
          throw RaisePyError(
              PyExc_RuntimeError,
              absl::StrCat("Unregistered proto message type: ", type_name)
                  .c_str());
        }
        // Creates a Packet with the concrete C++ payload type.
        std::unique_ptr<HolderBase> message_holder =
            std::move(maybe_holder).ValueOrDie();
        auto* copy = const_cast<proto_ns::MessageLite*>(
            message_holder->GetProtoMessageLite());
        copy->ParseFromString(std::string(serialized_proto));
        return packet_internal::Create(message_holder.release());
      },
      py::return_value_policy::move);

  m->def(
      "_create_proto_vector",
      [](const std::string& type_name,
         const std::vector<py::bytes>& serialized_proto_vector) {
        // TODO: Implement this.
        throw RaisePyError(PyExc_NotImplementedError,
                           "Creating a packet from a vector of proto messages "
                           "is not supproted yet.");
        return Packet();
      },
      py::return_value_policy::move);
}

void PacketCreatorSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule(
      "_packet_creator", "MediaPipe internal packet creator module.");
  PublicPacketCreators(&m);
  InternalPacketCreators(&m);
}

}  // namespace python
}  // namespace mediapipe
