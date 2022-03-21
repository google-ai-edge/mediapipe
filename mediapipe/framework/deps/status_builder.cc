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

#include "mediapipe/framework/deps/status_builder.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"

namespace mediapipe {

StatusBuilder::StatusBuilder(const StatusBuilder& sb) {
  status_ = sb.status_;
  file_ = sb.file_;
  line_ = sb.line_;
  no_logging_ = sb.no_logging_;
  stream_ = sb.stream_
                ? absl::make_unique<std::ostringstream>(sb.stream_->str())
                : nullptr;
  join_style_ = sb.join_style_;
}

StatusBuilder& StatusBuilder::operator=(const StatusBuilder& sb) {
  status_ = sb.status_;
  file_ = sb.file_;
  line_ = sb.line_;
  no_logging_ = sb.no_logging_;
  stream_ = sb.stream_
                ? absl::make_unique<std::ostringstream>(sb.stream_->str())
                : nullptr;
  join_style_ = sb.join_style_;
  return *this;
}

StatusBuilder& StatusBuilder::SetAppend() & {
  if (status_.ok()) return *this;
  join_style_ = MessageJoinStyle::kAppend;
  return *this;
}

StatusBuilder&& StatusBuilder::SetAppend() && { return std::move(SetAppend()); }

StatusBuilder& StatusBuilder::SetPrepend() & {
  if (status_.ok()) return *this;
  join_style_ = MessageJoinStyle::kPrepend;
  return *this;
}

StatusBuilder&& StatusBuilder::SetPrepend() && {
  return std::move(SetPrepend());
}

StatusBuilder& StatusBuilder::SetNoLogging() & {
  no_logging_ = true;
  return *this;
}

StatusBuilder&& StatusBuilder::SetNoLogging() && {
  return std::move(SetNoLogging());
}

StatusBuilder::operator Status() const& {
  if (!stream_ || stream_->str().empty() || no_logging_) {
    return status_;
  }
  return StatusBuilder(*this).JoinMessageToStatus();
}

StatusBuilder::operator Status() && {
  if (!stream_ || stream_->str().empty() || no_logging_) {
    return status_;
  }
  return JoinMessageToStatus();
}

absl::Status StatusBuilder::JoinMessageToStatus() {
  if (!stream_) {
    return absl::OkStatus();
  }
  std::string message;
  if (join_style_ == MessageJoinStyle::kAnnotate) {
    if (!status_.ok()) {
      message = absl::StrCat(status_.message(), "; ", stream_->str());
    }
  } else {
    message = join_style_ == MessageJoinStyle::kPrepend
                  ? absl::StrCat(stream_->str(), status_.message())
                  : absl::StrCat(status_.message(), stream_->str());
  }
  return Status(status_.code(), message);
}

}  // namespace mediapipe
