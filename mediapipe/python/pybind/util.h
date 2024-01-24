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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/timestamp.h"
#include "pybind11/gil.h"
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

inline void RaisePyErrorIfNotOk(const absl::Status& status,
                                bool acquire_gil = false) {
  if (!status.ok()) {
    if (acquire_gil) {
      py::gil_scoped_acquire acquire;
      throw RaisePyError(StatusCodeToPyError(status.code()),
                         status.message().data());
    } else {
      throw RaisePyError(StatusCodeToPyError(status.code()),
                         status.message().data());
    }
  }
}

inline void RaisePyErrorIfOverflow(int64_t value, int64_t min, int64_t max) {
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

// Reads a CalculatorGraphConfig from a file. If failed, raises a PyError.
inline ::mediapipe::CalculatorGraphConfig ReadCalculatorGraphConfigFromFile(
    const std::string& file_name) {
  ::mediapipe::CalculatorGraphConfig graph_config_proto;
  auto status = file::Exists(file_name);
  if (!status.ok()) {
    throw RaisePyError(PyExc_FileNotFoundError, status.message().data());
  }
  std::string graph_config_string;
  RaisePyErrorIfNotOk(file::GetContents(file_name, &graph_config_string,
                                        /*read_as_binary=*/true));
  if (!graph_config_proto.ParseFromArray(graph_config_string.c_str(),
                                         graph_config_string.length())) {
    throw RaisePyError(
        PyExc_RuntimeError,
        absl::StrCat("Failed to parse the binary graph: ", file_name).c_str());
  }
  return graph_config_proto;
}

}  // namespace python
}  // namespace mediapipe

#endif  // MEDIAPIPE_PYTHON_PYBIND_UTIL_H_
