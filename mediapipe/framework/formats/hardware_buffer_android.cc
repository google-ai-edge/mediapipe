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

#if (!defined(MEDIAPIPE_NO_JNI) ||                     \
     defined(MEDIAPIPE_ANDROID_LINK_NATIVE_WINDOW)) && \
    (__ANDROID_API__ >= 26 ||                          \
     defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))

#include <android/hardware_buffer.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/hardware_buffer.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

HardwareBuffer::HardwareBuffer(HardwareBuffer&& other) {
  spec_ = std::exchange(other.spec_, {});
  ahw_buffer_ = std::exchange(other.ahw_buffer_, nullptr);
  is_locked_ = std::exchange(other.is_locked_, false);
}

HardwareBuffer::HardwareBuffer(const HardwareBufferSpec& spec,
                               AHardwareBuffer* ahwb)
    : spec_(spec), ahw_buffer_(ahwb), is_locked_(false) {}

HardwareBuffer::~HardwareBuffer() { Reset(); }

absl::StatusOr<HardwareBuffer> HardwareBuffer::Create(
    const HardwareBufferSpec& spec) {
  MP_ASSIGN_OR_RETURN(AHardwareBuffer * ahwb, AllocateAHardwareBuffer(spec));
  return HardwareBuffer(spec, ahwb);
}

absl::StatusOr<HardwareBuffer> HardwareBuffer::WrapAndAcquireAHardwareBuffer(
    AHardwareBuffer* ahw_buffer) {
  MP_ASSIGN_OR_RETURN(HardwareBufferSpec spec,
                      AcquireAHardwareBuffer(ahw_buffer));
  return HardwareBuffer(spec, ahw_buffer);
}

bool HardwareBuffer::IsSupported() {
  if (__builtin_available(android 26, *)) {
    return true;
  }
  return false;
}

absl::StatusOr<AHardwareBuffer*> HardwareBuffer::AllocateAHardwareBuffer(
    const HardwareBufferSpec& spec) {
  RET_CHECK(IsSupported()) << "AndroidHWBuffers not supported";

  AHardwareBuffer* output = nullptr;
  int error = 0;
  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_Desc desc = {
        .width = spec.width,
        .height = spec.height,
        .layers = spec.layers,
        .format = spec.format,
        .usage = spec.usage,
    };
    error = AHardwareBuffer_allocate(&desc, &output);
  } else {
    return absl::UnavailableError(
        "NDK's hardware buffer support requires Android API level >= 26");
  }
  RET_CHECK(!error && output != nullptr)
      << "AHardwareBuffer_allocate failed: " << error;
  return output;
}

absl::StatusOr<HardwareBufferSpec> HardwareBuffer::AcquireAHardwareBuffer(
    AHardwareBuffer* ahw_buffer) {
  HardwareBufferSpec spec;
  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_Desc desc = {};
    AHardwareBuffer_describe(ahw_buffer, &desc);
    RET_CHECK_GT(desc.stride, 0) << "AHardwareBuffer_acquire failed";
    spec = {.width = desc.width,
            .height = desc.height,
            .layers = desc.layers,
            .format = desc.format,
            .usage = desc.usage,
            .stride = desc.stride};
    AHardwareBuffer_acquire(ahw_buffer);
  } else {
    return absl::UnavailableError(
        "NDK's hardware buffer support requires Android API level >= 26");
  }
  return spec;
}

absl::Status HardwareBuffer::ReleaseAHardwareBuffer() {
  if (ahw_buffer_ == nullptr) {
    return absl::OkStatus();
  }
  if (is_locked_) {
    MP_RETURN_IF_ERROR(Unlock());
  }
  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_release(ahw_buffer_);
  } else {
    return absl::UnavailableError(
        "NDK's hardware buffer support requires Android API level >= 26");
  }
  spec_ = {};
  ahw_buffer_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<void*> HardwareBuffer::Lock(
    uint64_t usage, std::optional<int> fence_file_descriptor) {
  RET_CHECK(ahw_buffer_ != nullptr) << "Hardware Buffer not allocated";
  RET_CHECK(!is_locked_) << "Hardware Buffer already locked";
  void* mem = nullptr;
  if (__builtin_available(android 26, *)) {
    const int error = AHardwareBuffer_lock(
        ahw_buffer_, usage,
        fence_file_descriptor.has_value() ? *fence_file_descriptor : -1,
        nullptr, &mem);
    RET_CHECK(error == 0) << "Hardware Buffer lock failed. Error: " << error;
  } else {
    return absl::UnavailableError(
        "NDK's hardware buffer support requires Android API level >= 26");
  }
  is_locked_ = true;
  return mem;
}

absl::Status HardwareBuffer::Unlock() {
  return UnlockInternal(/*fence_file_descriptor=*/nullptr);
}

absl::StatusOr<int> HardwareBuffer::UnlockAsync() {
  int fence_file_descriptor = -1;
  MP_RETURN_IF_ERROR(UnlockInternal(&fence_file_descriptor));
  return fence_file_descriptor;
}

absl::Status HardwareBuffer::UnlockInternal(int* fence_file_descriptor) {
  RET_CHECK(ahw_buffer_ != nullptr) << "Hardware Buffer not allocated";
  if (!is_locked_) {
    return absl::OkStatus();
  }
  if (__builtin_available(android 26, *)) {
    const int error =
        AHardwareBuffer_unlock(ahw_buffer_, fence_file_descriptor);
    RET_CHECK(error == 0) << "Hardware Buffer unlock failed. error: " << error;
  } else {
    return absl::UnavailableError(
        "NDK's hardware buffer support requires Android API level >= 26");
  }
  is_locked_ = false;
  return absl::OkStatus();
}

absl::StatusOr<uint32_t> HardwareBuffer::GetAlignedWidth() const {
  if (__builtin_available(android 26, *)) {
    RET_CHECK(ahw_buffer_ != nullptr) << "Hardware Buffer not allocated";
    AHardwareBuffer_Desc desc = {};
    AHardwareBuffer_describe(ahw_buffer_, &desc);
    RET_CHECK_GT(desc.stride, 0)
        << "AHardwareBuffer_describe failed, stride should not be zero";
    return desc.stride;
  }
  return absl::UnavailableError(
      "NDK's hardware buffer support requires Android API level >= 26");
}

void HardwareBuffer::Reset() {
  const auto success = ReleaseAHardwareBuffer();
  if (!success.ok()) {
    ABSL_LOG(DFATAL) << "Failed to release AHardwareBuffer: " << success;
  }
}

}  // namespace mediapipe

#endif  // (!defined(MEDIAPIPE_NO_JNI) ||
        // defined(MEDIAPIPE_ANDROID_LINK_NATIVE_WINDOW)) && (__ANDROID_API__>=
        // 26 || defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))
