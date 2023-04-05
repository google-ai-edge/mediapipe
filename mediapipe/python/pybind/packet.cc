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

#include "mediapipe/python/pybind/packet.h"

#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/python/pybind/util.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

void PacketSubmodule(pybind11::module* module) {
  py::module m = module->def_submodule("packet", "MediaPipe packet module.");

  py::class_<Packet> packet(
      m, "Packet",
      R"doc(The basic data flow unit of MediaPipe. A generic container class which can hold data of any type.

  A packet consists of a numeric timestamp and a shared pointer to an immutable
  payload. The payload can be of any C++ type (See packet_creator module for
  the list of the Python types that are supported). The payload's type is also
  referred to as the type of the packet. Packets are value classes and can be
  copied and moved cheaply. Each copy shares ownership of the payload, with be
  copied reference-counting semantics. Each copy has its own timestamp.

  The preferred method of creating a Packet is to invoke the methods in the
  "packet_creator" module. Packet contents can be retrieved by the methods in
  the "packet_getter" module.
)doc");

  packet.def(py::init(),
             R"doc(Create an empty Packet, for which is_empty() is True and
  timestamp() is Timestamp.unset. Calling packet getter methods on this Packet leads to runtime error.)doc");

  packet.def(
      "is_empty", &Packet::IsEmpty,
      R"doc(Return true iff the Packet has been created using the default constructor Packet(), or is a copy of such a Packet.)doc");

  packet.def(py::init<Packet const&>())
      .def("at", [](Packet* self,
                    int64_t ts_value) { return self->At(Timestamp(ts_value)); })
      .def("at", [](Packet* self, Timestamp ts) { return self->At(ts); })
      .def_property(
          "timestamp", &Packet::Timestamp,
          [](Packet* p, int64_t ts_value) { *p = p->At(Timestamp(ts_value)); })
      .def("__repr__", [](const Packet& self) {
        return absl::StrCat(
            "<mediapipe.Packet with timestamp: ",
            TimestampValueString(self.Timestamp()),
            self.IsEmpty()
                ? " and no data>"
                : absl::StrCat(" and C++ type: ", self.DebugTypeName(), ">"));
      });
}

}  // namespace python
}  // namespace mediapipe
