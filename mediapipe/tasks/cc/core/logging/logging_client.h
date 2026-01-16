// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_TASKS_CC_CORE_LOGGING_LOGGING_CLIENT_H_
#define MEDIAPIPE_TASKS_CC_CORE_LOGGING_LOGGING_CLIENT_H_

#include "mediapipe/util/analytics/mediapipe_log_extension.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace logging {

class LoggingClient {
 public:
  virtual ~LoggingClient() = default;
  virtual void LogEvent(
      const logs::proto::mediapipe::MediaPipeLogExtension& log) = 0;
};

}  // namespace logging
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_LOGGING_LOGGING_CLIENT_H_
