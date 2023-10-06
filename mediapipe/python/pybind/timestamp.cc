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

#include "mediapipe/python/pybind/timestamp.h"

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/python/pybind/util.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

void TimestampSubmodule(pybind11::module* module) {
  py::module m =
      module->def_submodule("timestamp", "MediaPipe timestamp module.");

  py::class_<Timestamp> timestamp(
      m, "Timestamp",
      R"doc(A class which represents a timestamp in the MediaPipe framework.

  MediaPipe timestamps are in units of _microseconds_.
  There are several special values (All these values must be constructed using
  the static methods provided):
    UNSET:       The default initialization value, not generally valid when a
                 timestamp is required.
    UNSTARTED:   The timestamp before any valid timestamps. This is the input
                 timestamp during Calcultor::Open().
    PRESTREAM:   A value for specifying that a packet contains "header" data
                 that should be processed before any other timestamp.  Like
                 poststream, if this value is sent then it must be the only
                 value that is sent on the stream.
    MIN:         The minimum range timestamp to see in Calcultor::Process().
                 Any number of "range" timestamp can be sent over a stream,
                 provided that they are sent in monotonically increasing order.
    MAX:         The maximum range timestamp to see in Process().
    POSTSTREAM:  A value for specifying that a packet pertains to the entire
                 stream.  This "summary" timestamp occurs after all the "range"
                 timestamps.  If this timestamp is sent on a stream, it must be
                 the only packet sent.
    DONE:        The timestamp after all valid timestamps.
                 This is the input timestamp during Calcultor::Close().
)doc");

  timestamp.def(py::init<const Timestamp&>())
      .def(py::init<int64_t>())
      .def_property_readonly("value", &Timestamp::Value)
      .def_property_readonly_static(
          "UNSET", [](py::object) { return Timestamp::Unset(); })
      .def_property_readonly_static(
          "UNSTARTED", [](py::object) { return Timestamp::Unstarted(); })
      .def_property_readonly_static(
          "PRESTREAM", [](py::object) { return Timestamp::PreStream(); })
      .def_property_readonly_static("MIN",
                                    [](py::object) { return Timestamp::Min(); })
      .def_property_readonly_static("MAX",
                                    [](py::object) { return Timestamp::Max(); })
      .def_property_readonly_static(
          "POSTSTREAM", [](py::object) { return Timestamp::PostStream(); })
      .def_property_readonly_static(
          "DONE", [](py::object) { return Timestamp::Done(); })
      .def("__eq__",
           [](const Timestamp& a, const Timestamp& b) { return a == b; })
      .def("__lt__",
           [](const Timestamp& a, const Timestamp& b) { return a < b; })
      .def("__gt__",
           [](const Timestamp& a, const Timestamp& b) { return a > b; })
      .def("__le__",
           [](const Timestamp& a, const Timestamp& b) { return a <= b; })
      .def("__ge__",
           [](const Timestamp& a, const Timestamp& b) { return a >= b; })
      .def("__repr__", [](const Timestamp& self) {
        return absl::StrCat("<mediapipe.Timestamp with value: ",
                            TimestampValueString(self), ">");
      });

  timestamp.def("seconds", &Timestamp::Seconds,
                R"doc(Return the value in units of seconds as a float.)doc");

  timestamp.def(
      "microseconds", &Timestamp::Microseconds,
      R"doc(Return the value in units of microseconds as an int.)doc");

  timestamp.def("is_special_value", &Timestamp::IsSpecialValue,
                R"doc(Check if the timestamp is a special value,

    A special value is any of the values which cannot be constructed directly
    but must be constructed using the static special value.

)doc");

  timestamp.def(
      "is_range_value", &Timestamp::IsRangeValue,
      R"doc(Check if the timestamp is a range value is anything between Min() and Max() (inclusive).

  Any number of packets with range values can be sent over a stream as long as
  they are sent in monotonically increasing order. is_range_value() isn't
  quite the opposite of is_special_value() since it is valid to start a stream
  at Timestamp::Min() and continue until timestamp max (both of which are
  special values). prestream and postStream  are not considered a range value
  even though they can be sent over a stream (they are "summary" timestamps not
  "range" timestamps).
)doc");

  timestamp.def(
      "is_allowed_in_stream", &Timestamp::IsAllowedInStream,
      R"doc(Returns true iff this can be the timestamp of a Packet in a stream.

  Any number of RangeValue timestamps may be in a stream (in monotonically
  increasing order).  Also, exactly one prestream, or one poststream packet is
  allowed.
)doc");

  timestamp.def_static("from_seconds", &Timestamp::FromSeconds,
                       R"doc(Create a timestamp from a seconds value

  Args:
    seconds: A seconds value in float.

  Returns:
    A MediaPipe Timestamp object.

  Examples:
    timestamp_now = mp.Timestamp.from_seconds(time.time())
)doc");

  py::implicitly_convertible<int64_t, Timestamp>();
}

}  // namespace python
}  // namespace mediapipe
