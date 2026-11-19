// Copyright 2026 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_TASKS_C_CORE_RESULT_CALLBACK_H_
#define MEDIAPIPE_TASKS_C_CORE_RESULT_CALLBACK_H_

#include <string>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/log/absl_check.h"   // from @com_google_absl
#include "absl/status/status.h"    // from @com_google_absl
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"

namespace mediapipe::tasks::c::core {

// Calls the result callback of a C task with the `user_data` that the caller
// set next to the callback in the options of the task.
//
// The result callbacks of the C API have the form
//
//   void Callback(void* user_data, MpStatus status, const char* error_msg,
//                 const Result* result, Args... args);
//
// where `Result` is the result struct of the task and `Args` are the arguments
// that the task reports with every result, e.g. the image and the timestamp of
// a vision task.
//
// Example:
//
//   ResultCallback callback(options.result_callback,
//                           options.result_callback_user_data);
//   cpp_options->result_callback = [callback](...) {
//     if (!cpp_result.ok()) {
//       callback.InvokeWithError(cpp_result.status(), &image, timestamp_ms);
//       return;
//     }
//     ...
//     callback.InvokeWithResult(&result, &image, timestamp_ms);
//   };
template <typename Result, typename... Args>
class ResultCallback {
 public:
  ResultCallback(void (*callback)(void* user_data, MpStatus status,
                                  const char* error_msg, const Result* result,
                                  Args... args),
                 void* user_data)
      : callback_(callback), user_data_(user_data) {
    ABSL_DCHECK(callback_ != nullptr);
  }

  // Reports `result` with the status `kMpOk` and without an error message.
  //
  // CFI is disabled for the call because the callback is provided by the user
  // of the C API, who may have built it without CFI.
  ABSL_ATTRIBUTE_NO_SANITIZE_CFI void InvokeWithResult(const Result* result,
                                                       Args... args) const {
    callback_(user_data_, kMpOk, /*error_msg=*/nullptr, result, args...);
  }

  // Reports the error `status`, which must not be OK, with its message and
  // without a result. The message is only valid during the call.
  ABSL_ATTRIBUTE_NO_SANITIZE_CFI void InvokeWithError(
      const absl::Status& status, Args... args) const {
    ABSL_DCHECK(!status.ok());
    const std::string error_msg(status.message());
    callback_(user_data_, ToMpStatus(status), error_msg.c_str(),
              /*result=*/nullptr, args...);
  }

 private:
  void (*callback_)(void*, MpStatus, const char*, const Result*, Args...);
  void* user_data_;
};

// Deduces `Result` and `Args` from the type of the C callback.
template <typename Result, typename... Args>
ResultCallback(void (*)(void*, MpStatus, const char*, const Result*, Args...),
               void*) -> ResultCallback<Result, Args...>;

}  // namespace mediapipe::tasks::c::core

#endif  // MEDIAPIPE_TASKS_C_CORE_RESULT_CALLBACK_H_
