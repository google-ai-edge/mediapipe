// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/deps/status.h"

#include <stdio.h>

namespace mediapipe {

Status::Status(::mediapipe::StatusCode code, absl::string_view msg) {
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string& Status::empty_string() {
  static std::string* empty = new std::string;
  return *empty;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    char tmp[30];
    const char* type;
    switch (code()) {
      case ::mediapipe::StatusCode::kCancelled:
        type = "Cancelled";
        break;
      case ::mediapipe::StatusCode::kUnknown:
        type = "Unknown";
        break;
      case ::mediapipe::StatusCode::kInvalidArgument:
        type = "Invalid argument";
        break;
      case ::mediapipe::StatusCode::kDeadlineExceeded:
        type = "Deadline exceeded";
        break;
      case ::mediapipe::StatusCode::kNotFound:
        type = "Not found";
        break;
      case ::mediapipe::StatusCode::kAlreadyExists:
        type = "Already exists";
        break;
      case ::mediapipe::StatusCode::kPermissionDenied:
        type = "Permission denied";
        break;
      case ::mediapipe::StatusCode::kUnauthenticated:
        type = "Unauthenticated";
        break;
      case ::mediapipe::StatusCode::kResourceExhausted:
        type = "Resource exhausted";
        break;
      case ::mediapipe::StatusCode::kFailedPrecondition:
        type = "Failed precondition";
        break;
      case ::mediapipe::StatusCode::kAborted:
        type = "Aborted";
        break;
      case ::mediapipe::StatusCode::kOutOfRange:
        type = "Out of range";
        break;
      case ::mediapipe::StatusCode::kUnimplemented:
        type = "Unimplemented";
        break;
      case ::mediapipe::StatusCode::kInternal:
        type = "Internal";
        break;
      case ::mediapipe::StatusCode::kUnavailable:
        type = "Unavailable";
        break;
      case ::mediapipe::StatusCode::kDataLoss:
        type = "Data loss";
        break;
      default:
        snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                 static_cast<int>(code()));
        type = tmp;
        break;
    }
    std::string result(type);
    result += ": ";
    result += state_->msg;
    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

std::string* MediaPipeCheckOpHelperOutOfLine(const ::mediapipe::Status& v,
                                             const char* msg) {
  std::string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks std::string but this is only to be used in a fatal error message
  return new std::string(r);
}

}  // namespace mediapipe
