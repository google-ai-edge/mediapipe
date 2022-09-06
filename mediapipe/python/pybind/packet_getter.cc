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

#include "mediapipe/python/pybind/packet_getter.h"

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
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

template <typename T>
const T& GetContent(const Packet& packet) {
  RaisePyErrorIfNotOk(packet.ValidateAsType<T>());
  return packet.Get<T>();
}

}  // namespace

namespace py = pybind11;

void PublicPacketGetters(pybind11::module* m) {
  m->def("get_str", &GetContent<std::string>,
         R"doc(Get the content of a MediaPipe string Packet as a str.

  Args:
    packet: A MediaPipe string Packet.

  Returns:
    A str.

  Raises:
    ValueError: If the Packet doesn't contain string data.

  Examples:
    packet = mp.packet_creator.create_string('abc')
    data = mp.packet_getter.get_str(packet)
)doc");

  m->def(
      "get_bytes",
      [](const Packet& packet) {
        return py::bytes(GetContent<std::string>(packet));
      },
      R"doc(Get the content of a MediaPipe string Packet as a bytes object.

  Args:
    packet: A MediaPipe string Packet.

  Returns:
    A bytes object.

  Raises:
    ValueError: If the Packet doesn't contain string data.

  Examples:
    packet = mp.packet_creator.create_string(b'\xd0\xd0\xd0')
    data = mp.packet_getter.get_bytes(packet)
)doc");

  m->def("get_bool", &GetContent<bool>,
         R"doc(Get the content of a MediaPipe bool Packet as a boolean.

  Args:
    packet: A MediaPipe bool Packet.

  Returns:
    A boolean.

  Raises:
    ValueError: If the Packet doesn't contain bool data.

  Examples:
    packet = mp.packet_creator.create_bool(True)
    data = mp.packet_getter.get_bool(packet)
)doc");

  m->def(
      "get_int",
      [](const Packet& packet) {
        if (packet.ValidateAsType<int>().ok()) {
          return static_cast<int64>(packet.Get<int>());
        } else if (packet.ValidateAsType<int8>().ok()) {
          return static_cast<int64>(packet.Get<int8>());
        } else if (packet.ValidateAsType<int16>().ok()) {
          return static_cast<int64>(packet.Get<int16>());
        } else if (packet.ValidateAsType<int32>().ok()) {
          return static_cast<int64>(packet.Get<int32>());
        } else if (packet.ValidateAsType<int64>().ok()) {
          return static_cast<int64>(packet.Get<int64>());
        }
        throw RaisePyError(
            PyExc_ValueError,
            "Packet doesn't contain int, int8, int16, int32, or int64 data.");
      },
      R"doc(Get the content of a MediaPipe int Packet as an integer.

  Args:
    packet: A MediaPipe Packet that holds int, int8, int16, int32, or int64 data.

  Returns:
    An integer.

  Raises:
    ValueError: If the Packet doesn't contain int, int8, int16, int32, or int64 data.

  Examples:
    packet = mp.packet_creator.create_int(0)
    data = mp.packet_getter.get_int(packet)
)doc");

  m->def(
      "get_uint",
      [](const Packet& packet) {
        if (packet.ValidateAsType<uint8>().ok()) {
          return static_cast<std::uint64_t>(packet.Get<uint8>());
        } else if (packet.ValidateAsType<uint16>().ok()) {
          return static_cast<std::uint64_t>(packet.Get<uint16>());
        } else if (packet.ValidateAsType<uint32>().ok()) {
          return static_cast<std::uint64_t>(packet.Get<uint32>());
        } else if (packet.ValidateAsType<uint64>().ok()) {
          return static_cast<std::uint64_t>(packet.Get<uint64>());
        }
        throw RaisePyError(
            PyExc_ValueError,
            "Packet doesn't contain uint8, uint16, uint32, or uint64 data.");
      },
      R"doc(Get the content of a MediaPipe uint Packet as an integer.

  Args:
    packet: A MediaPipe Packet that holds uint8, uint16, uint32, or uint64 data.

  Raises:
    ValueError: If the Packet doesn't contain uint8, uint16, uint32, or uint64 data.

  Returns:
    An integer.

  Examples:
    packet = mp.packet_creator.create_uint8(2**8 - 1)
    data = mp.packet_getter.get_uint(packet)
)doc");

  m->def(
      "get_float",
      [](const Packet& packet) {
        if (packet.ValidateAsType<float>().ok()) {
          return packet.Get<float>();
        } else if (packet.ValidateAsType<double>().ok()) {
          return static_cast<float>(packet.Get<double>());
        }
        throw RaisePyError(PyExc_ValueError,
                           "Packet doesn't contain float or double data.");
      },
      R"doc(Get the content of a MediaPipe float or double Packet as a float.

  Args:
    packet: A MediaPipe Packet that holds float or double data.

  Raises:
    ValueError: If the Packet doesn't contain float or double data.

  Returns:
    A float.

  Examples:
    packet = mp.packet_creator.create_float(0.1)
    data = mp.packet_getter.get_float(packet)
)doc");

  m->def(
      "get_int_list",
      [](const Packet& packet) {
        if (packet.ValidateAsType<std::vector<int>>().ok()) {
          auto int_list = packet.Get<std::vector<int>>();
          return std::vector<int64>(int_list.begin(), int_list.end());
        } else if (packet.ValidateAsType<std::vector<int8>>().ok()) {
          auto int_list = packet.Get<std::vector<int8>>();
          return std::vector<int64>(int_list.begin(), int_list.end());
        } else if (packet.ValidateAsType<std::vector<int16>>().ok()) {
          auto int_list = packet.Get<std::vector<int16>>();
          return std::vector<int64>(int_list.begin(), int_list.end());
        } else if (packet.ValidateAsType<std::vector<int32>>().ok()) {
          auto int_list = packet.Get<std::vector<int32>>();
          return std::vector<int64>(int_list.begin(), int_list.end());
        } else if (packet.ValidateAsType<std::vector<int64>>().ok()) {
          auto int_list = packet.Get<std::vector<int64>>();
          return std::vector<int64>(int_list.begin(), int_list.end());
        }
        throw RaisePyError(PyExc_ValueError,
                           "Packet doesn't contain int, int8, int16, int32, or "
                           "int64 containers.");
      },
      R"doc(Get the content of a MediaPipe int vector Packet as an integer list.

  Args:
    packet: A MediaPipe Packet that holds std:vector<int>.

  Returns:
    An integer list.

  Raises:
    ValueError: If the Packet doesn't contain std:vector<int>.

  Examples:
    packet = mp.packet_creator.create_int_vector([1, 2, 3])
    data = mp.packet_getter.get_int_list(packet)
)doc");

  m->def(
      "get_bool_list", &GetContent<std::vector<bool>>,
      R"doc(Get the content of a MediaPipe bool vector Packet as a boolean list.

  Args:
    packet: A MediaPipe Packet that holds std:vector<bool>.

  Returns:
    An boolean list.

  Raises:
    ValueError: If the Packet doesn't contain std:vector<bool>.

  Examples:
    packet = mp.packet_creator.create_bool_vector([True, True, False])
    data = mp.packet_getter.get_bool_list(packet)
)doc");

  m->def(
      "get_float_list",
      [](const Packet& packet) {
        if (packet.ValidateAsType<std::vector<float>>().ok()) {
          return packet.Get<std::vector<float>>();
        } else if (packet.ValidateAsType<std::array<float, 16>>().ok()) {
          auto float_array = packet.Get<std::array<float, 16>>();
          return std::vector<float>(float_array.begin(), float_array.end());
        } else if (packet.ValidateAsType<std::array<float, 4>>().ok()) {
          auto float_array = packet.Get<std::array<float, 4>>();
          return std::vector<float>(float_array.begin(), float_array.end());
        } else {
          throw RaisePyError(PyExc_ValueError,
                             "Packet doesn't contain std::vector<float> or "
                             "std::array<float, 4 / 16> containers.");
        }
      },
      R"doc(Get the content of a MediaPipe float vector Packet as a float list.

  Args:
    packet: A MediaPipe Packet that holds std:vector<float>.

  Returns:
    A float list.

  Raises:
    ValueError: If the Packet doesn't contain std:vector<float>.

  Examples:
    packet = packet_creator.create_float_vector([0.1, 0.2, 0.3])
    data = packet_getter.get_float_list(packet)
)doc");

  m->def(
      "get_str_list", &GetContent<std::vector<std::string>>,
      R"doc(Get the content of a MediaPipe string vector Packet as a str list.

  Args:
    packet: A MediaPipe Packet that holds std:vector<std::string>.

  Returns:
    A str list.

  Raises:
    ValueError: If the Packet doesn't contain std:vector<std::string>.

  Examples:
    packet = mp.packet_creator.create_string_vector(['a', 'b', 'c'])
    data = mp.packet_getter.get_str_list(packet)
)doc");

  m->def(
      "get_image_list", &GetContent<std::vector<Image>>,
      R"doc(Get the content of a MediaPipe Packet of image vector as a list of MediaPipe Images.

  Args:
    packet: A MediaPipe Packet that holds std:vector<mediapipe::Image>.

  Returns:
    A list of MediaPipe Images.

  Raises:
    ValueError: If the Packet doesn't contain std:vector<mediapipe::Image>.

  Examples:
    packet = mp.packet_creator.create_image_vector([
        image1, image2, image3])
    image_list = mp.packet_getter.get_image_list(packet)
)doc");

  m->def(
      "get_packet_list", &GetContent<std::vector<Packet>>,
      R"doc(Get the content of a MediaPipe Packet of Packet vector as a Packet list.

  Args:
    packet: A MediaPipe Packet that holds std:vector<Packet>.

  Returns:
    A Packet list.

  Raises:
    ValueError: If the Packet doesn't contain std:vector<Packet>.

  Examples:
    packet = mp.packet_creator.create_packet_vector([
        packet_creator.create_float(0.1),
        packet_creator.create_int(1),
        packet_creator.create_string('1')
    ])
    packet_list = mp.packet_getter.get_packet_list(packet)
)doc");

  m->def(
      "get_str_to_packet_dict", &GetContent<std::map<std::string, Packet>>,

      R"doc(Get the content of a MediaPipe Packet as a dictionary that has (str, Packet) pairs.

  Args:
    packet: A MediaPipe Packet that holds std::map<std::string, Packet>.

  Returns:
    A dictionary that has (str, Packet) pairs.

  Raises:
    ValueError: If the Packet doesn't contain std::map<std::string, Packet>.

  Examples:
    dict_packet = mp.packet_creator.create_string_to_packet_map({
        'float': packet_creator.create_float(0.1),
        'int': packet_creator.create_int(1),
        'string': packet_creator.create_string('1')
    data = mp.packet_getter.get_str_to_packet_dict(dict_packet)
)doc");

  m->def(
      "get_image_frame", &GetContent<ImageFrame>,
      R"doc(Get the content of a MediaPipe ImageFrame Packet as an ImageFrame object.

  Args:
    packet: A MediaPipe ImageFrame Packet.

  Returns:
    A MediaPipe ImageFrame object.

  Raises:
    ValueError: If the Packet doesn't contain ImageFrame.

  Examples:
    packet = packet_creator.create_image_frame(frame)
    data = packet_getter.get_image_frame(packet)
)doc",
      py::return_value_policy::reference_internal);

  m->def("get_image", &GetContent<Image>,
         R"doc(Get the content of a MediaPipe Image Packet as an Image object.

  Args:
    packet: A MediaPipe Image Packet.

  Returns:
    A MediaPipe Image object.

  Raises:
    ValueError: If the Packet doesn't contain Image.

  Examples:
    packet = packet_creator.create_image(frame)
    data = packet_getter.get_image(packet)
)doc",
         py::return_value_policy::reference_internal);

  m->def(
      "get_matrix",
      [](const Packet& packet) {
        return Eigen::Ref<const Eigen::MatrixXf>(GetContent<Matrix>(packet));
      },
      R"doc(Get the content of a MediaPipe Matrix Packet as a numpy 2d float ndarray.

  Args:
    packet: A MediaPipe Matrix Packet.

  Returns:
    A numpy 2d float ndarray.

  Raises:
    ValueError: If the Packet doesn't contain matrix data.

  Examples:
    packet = mp.packet_creator.create_matrix(2d_array)
    data = mp.packet_getter.get_matrix(packet)
)doc",
      py::return_value_policy::reference_internal);
}

void InternalPacketGetters(pybind11::module* m) {
  m->def(
      "_get_proto_type_name",
      [](const Packet& packet) {
        return packet.GetProtoMessageLite().GetTypeName();
      },
      py::return_value_policy::move);

  m->def(
      "_get_proto_vector_size",
      [](Packet& packet) {
        auto proto_vector = packet.GetVectorOfProtoMessageLitePtrs();
        RaisePyErrorIfNotOk(proto_vector.status());
        return proto_vector.value().size();
      },
      py::return_value_policy::move);

  m->def(
      "_get_proto_vector_element_type_name",
      [](Packet& packet) {
        auto proto_vector = packet.GetVectorOfProtoMessageLitePtrs();
        RaisePyErrorIfNotOk(proto_vector.status());
        if (proto_vector.value().empty()) {
          return std::string();
        }
        return proto_vector.value()[0]->GetTypeName();
      },
      py::return_value_policy::move);

  m->def(
      "_get_serialized_proto",
      [](const Packet& packet) {
        // By default, py::bytes is an extra copy of the original string object:
        // https://github.com/pybind/pybind11/issues/1236
        // However, when Pybind11 performs the C++ to Python transition, it
        // only increases the py::bytes object's ref count. See the
        // implmentation at line 1583 in "pybind11/cast.h".
        return py::bytes(packet.GetProtoMessageLite().SerializeAsString());
      },
      py::return_value_policy::move);

  m->def(
      "_get_serialized_proto_list",
      [](Packet& packet) {
        auto proto_vector = packet.GetVectorOfProtoMessageLitePtrs();
        RaisePyErrorIfNotOk(proto_vector.status());
        int size = proto_vector.value().size();
        std::vector<py::bytes> results;
        results.reserve(size);
        for (const proto_ns::MessageLite* ptr : proto_vector.value()) {
          results.push_back(py::bytes(ptr->SerializeAsString()));
        }
        return results;
      },
      py::return_value_policy::move);
}

void PacketGetterSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule(
      "_packet_getter", "MediaPipe internal packet getter module.");
  PublicPacketGetters(&m);
  InternalPacketGetters(&m);
}

}  // namespace python
}  // namespace mediapipe
