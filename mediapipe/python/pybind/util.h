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

#ifndef MEDIAPIPE_PYTHON_PYBIND_UTIL_H_
#define MEDIAPIPE_PYTHON_PYBIND_UTIL_H_

#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace python {

namespace py = pybind11;

inline py::error_already_set RaisePyError(PyObject* exc_class,
                                          const char* message) {
  PyErr_SetString(exc_class, message);
  return py::error_already_set();
}

inline PyObject* StatusCodeToPyError(const ::absl::StatusCode& code) {
  switch (code) {
    case absl::StatusCode::kInvalidArgument:
      return static_cast<PyObject*>(PyExc_ValueError);
    case absl::StatusCode::kAlreadyExists:
      return static_cast<PyObject*>(PyExc_FileExistsError);
    case absl::StatusCode::kUnimplemented:
      return static_cast<PyObject*>(PyExc_NotImplementedError);
    default:
      return static_cast<PyObject*>(PyExc_RuntimeError);
  }
}

inline void RaisePyErrorIfNotOk(const mediapipe::Status& status) {
  if (!status.ok()) {
    throw RaisePyError(StatusCodeToPyError(status.code()),
                       status.message().data());
  }
}

inline void RaisePyErrorIfOverflow(int64 value, int64 min, int64 max) {
  if (value > max) {
    throw RaisePyError(PyExc_OverflowError,
                       absl::StrCat(value, " execeeds the maximum value (", max,
                                    ") the data type can have.")
                           .c_str());
  } else if (value < min) {
    throw RaisePyError(PyExc_OverflowError,
                       absl::StrCat(value, " goes below the minimum value (",
                                    min, ") the data type can have.")
                           .c_str());
  }
}

inline std::string TimestampValueString(const Timestamp& timestamp) {
  if (timestamp == Timestamp::Unset()) {
    return "UNSET";
  } else if (timestamp == Timestamp::Unstarted()) {
    return "UNSTARTED";
  } else if (timestamp == Timestamp::PreStream()) {
    return "PRESTREAM";
  } else if (timestamp == Timestamp::Min()) {
    return "MIN";
  } else if (timestamp == Timestamp::Max()) {
    return "MAX";
  } else if (timestamp == Timestamp::PostStream()) {
    return "POSTSTREAM";
  } else if (timestamp == Timestamp::OneOverPostStream()) {
    return "ONEOVERPOSTSTREAM";
  } else if (timestamp == Timestamp::Done()) {
    return "DONE";
  } else {
    return timestamp.DebugString();
  }
}

}  // namespace python
}  // namespace mediapipe

#endif  // MEDIAPIPE_PYTHON_PYBIND_UTIL_H_
