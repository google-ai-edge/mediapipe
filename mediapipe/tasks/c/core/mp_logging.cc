/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/c/core/mp_logging.h"

#include <string>

#include "absl/base/const_init.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/synchronization/mutex.h"

namespace {

class CCallbackLogSink : public absl::LogSink {
 public:
  CCallbackLogSink(MpLogCallback callback, void* user_data)
      : callback_(callback), user_data_(user_data) {}

  void Send(const absl::LogEntry& entry) override {
    if (callback_) {
      std::string msg(entry.text_message());
      callback_(static_cast<int>(entry.log_severity()), msg.c_str(),
                user_data_);
    }
  }

 private:
  MpLogCallback callback_;
  void* user_data_;
};

CCallbackLogSink* g_log_sink = nullptr;
absl::Mutex g_log_sink_mutex(absl::kConstInit);

}  // namespace

extern "C" {

void MpSetLoggingCallback(MpLogCallback callback, void* user_data) {
  if (callback == nullptr) {
    MpRemoveLoggingCallback();
    return;
  }
  absl::MutexLock lock(g_log_sink_mutex);
  if (g_log_sink) {
    absl::RemoveLogSink(g_log_sink);
    delete g_log_sink;
    g_log_sink = nullptr;
  }
  g_log_sink = new CCallbackLogSink(callback, user_data);
  absl::AddLogSink(g_log_sink);
}

void MpRemoveLoggingCallback(void) {
  absl::MutexLock lock(g_log_sink_mutex);
  if (g_log_sink) {
    absl::RemoveLogSink(g_log_sink);
    delete g_log_sink;
    g_log_sink = nullptr;
  }
}

}  // extern "C"
