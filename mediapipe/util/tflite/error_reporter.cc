// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/util/tflite/error_reporter.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include "tensorflow/lite/minimal_logging.h"

namespace mediapipe {
namespace util {
namespace tflite {

ErrorReporter::ErrorReporter() {
  message_[0] = '\0';
  previous_message_[0] = '\0';
}

int ErrorReporter::Report(const char* format, va_list args) {
  std::strcpy(previous_message_, message_);  // NOLINT
  message_[0] = '\0';
  int num_characters = vsnprintf(message_, kBufferSize, format, args);
  // To mimic tflite::StderrReporter.
  ::tflite::logging_internal::MinimalLogger::Log(::tflite::TFLITE_LOG_ERROR,
                                                 "%s", message_);
  return num_characters;
}

bool ErrorReporter::HasError() const { return message_[0] != '\0'; }

std::string ErrorReporter::message() { return message_; }

std::string ErrorReporter::previous_message() { return previous_message_; }

}  // namespace tflite
}  // namespace util
}  // namespace mediapipe
