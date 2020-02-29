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

#ifndef MEDIAPIPE_DEPS_STATUS_H_
#define MEDIAPIPE_DEPS_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

using Status = absl::Status;
using StatusCode = absl::StatusCode;

inline ::mediapipe::Status OkStatus() { return absl::OkStatus(); }

extern std::string* MediaPipeCheckOpHelperOutOfLine(
    const ::mediapipe::Status& v, const char* msg);

inline std::string* MediaPipeCheckOpHelper(::mediapipe::Status v,
                                           const char* msg) {
  if (v.ok()) return nullptr;
  return MediaPipeCheckOpHelperOutOfLine(v, msg);
}

#define MEDIAPIPE_DO_CHECK_OK(val, level)                               \
  while (auto _result = ::mediapipe::MediaPipeCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

// To be consistent with MP_EXPECT_OK, we add prefix MEDIAPIPE_ to
// CHECK_OK, QCHECK_OK, and DCHECK_OK. We prefer to use the marcos with
// MEDIAPIPE_ prefix in mediapipe's codebase.
#define MEDIAPIPE_CHECK_OK(val) MEDIAPIPE_DO_CHECK_OK(val, FATAL)
#define MEDIAPIPE_QCHECK_OK(val) MEDIAPIPE_DO_CHECK_OK(val, QFATAL)

#ifndef NDEBUG
#define MEDIAPIPE_DCHECK_OK(val) MEDIAPIPE_CHECK_OK(val)
#else
#define MEDIAPIPE_DCHECK_OK(val) \
  while (false && (::mediapipe::OkStatus() == (val))) LOG(FATAL)
#endif

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_STATUS_H_
