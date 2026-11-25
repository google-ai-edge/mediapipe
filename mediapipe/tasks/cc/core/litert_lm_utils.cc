/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/core/litert_lm_utils.h"

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>

#include <cerrno>
#endif

#include <memory>
#include <string>

#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "absl/strings/str_cat.h"       // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "mediapipe/tasks/cc/core/base_options.h"
#include "runtime/executor/executor_settings_base.h"  // from @litert_lm
#include "runtime/util/scoped_file.h"                 // from @litert_lm

namespace mediapipe::tasks::core {

litert::lm::Backend ToLiteRtLmBackend(BaseOptions::Delegate delegate) {
  switch (delegate) {
    case BaseOptions::Delegate::GPU:
      return litert::lm::Backend::GPU;
    case BaseOptions::Delegate::NPU:
      return litert::lm::Backend::NPU;
    default:
      return litert::lm::Backend::CPU;
  }
}

absl::Status ValidateLiteRtLmModelAsset(const BaseOptions& base_options,
                                        absl::string_view option_name) {
  if (base_options.model_asset_buffer != nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        option_name,
        " does not support in-memory model_asset_buffer; provide "
        "model_asset_path or model_asset_descriptor_meta.fd instead."));
  }
  if (base_options.model_asset_path.empty() &&
      base_options.model_asset_descriptor_meta.fd < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Either model_asset_path or model_asset_descriptor_meta.fd"
                     " must be specified in ",
                     option_name, "."));
  }
  if (!base_options.model_asset_path.empty() &&
      base_options.model_asset_descriptor_meta.fd >= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Only one of model_asset_path or model_asset_descriptor_meta.fd may be "
        "specified in ",
        option_name, "."));
  }
  if (base_options.model_asset_descriptor_meta.fd >= 0) {
    const auto& fd_meta = base_options.model_asset_descriptor_meta;
    if (fd_meta.offset != -1) {
      return absl::InvalidArgumentError(
          absl::StrCat(option_name,
                       ".model_asset_descriptor_meta.offset is not supported; "
                       "provide a whole-file descriptor."));
    }
    if (fd_meta.length != -1) {
      return absl::InvalidArgumentError(
          absl::StrCat(option_name,
                       ".model_asset_descriptor_meta.length is not supported; "
                       "provide a whole-file descriptor."));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<litert::lm::ModelAssets> CreateLiteRtLmModelAssets(
    const BaseOptions& base_options, absl::string_view option_name) {
  ABSL_RETURN_IF_ERROR(ValidateLiteRtLmModelAsset(base_options, option_name));
  const auto& fd_meta = base_options.model_asset_descriptor_meta;
  if (fd_meta.fd < 0) {
    return litert::lm::ModelAssets::Create(base_options.model_asset_path);
  }
#ifdef _WIN32
  const HANDLE source_handle =
      reinterpret_cast<HANDLE>(_get_osfhandle(fd_meta.fd));
  if (source_handle == INVALID_HANDLE_VALUE) {
    return absl::InvalidArgumentError(absl::StrCat(
        option_name,
        ".model_asset_descriptor_meta.fd is not a valid file descriptor."));
  }
  HANDLE duplicate_handle = INVALID_HANDLE_VALUE;
  if (!DuplicateHandle(GetCurrentProcess(), source_handle, GetCurrentProcess(),
                       &duplicate_handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    const DWORD error = GetLastError();
    const std::string message = absl::StrCat(
        "Failed to duplicate ", option_name,
        ".model_asset_descriptor_meta.fd (Win32 error ", error, ").");
    if (error == ERROR_INVALID_HANDLE) {
      return absl::InvalidArgumentError(message);
    }
    if (error == ERROR_TOO_MANY_OPEN_FILES ||
        error == ERROR_NOT_ENOUGH_MEMORY || error == ERROR_OUTOFMEMORY) {
      return absl::ResourceExhaustedError(message);
    }
    return absl::InternalError(message);
  }
  return litert::lm::ModelAssets::Create(
      std::make_shared<litert::lm::ScopedFile>(duplicate_handle));
#else
  const int duplicate_fd = fcntl(fd_meta.fd, F_DUPFD_CLOEXEC, 0);
  if (duplicate_fd < 0) {
    const int err = errno;
    const absl::Status status = absl::ErrnoToStatus(
        err, absl::StrCat("Failed to duplicate ", option_name,
                          ".model_asset_descriptor_meta.fd"));
    if (err == EBADF) {
      return absl::InvalidArgumentError(status.message());
    }
    return status;
  }
  return litert::lm::ModelAssets::Create(
      std::make_shared<litert::lm::ScopedFile>(duplicate_fd));
#endif
}

}  // namespace mediapipe::tasks::core
