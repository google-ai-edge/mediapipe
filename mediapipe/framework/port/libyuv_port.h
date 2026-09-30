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

#ifndef MEDIAPIPE_FRAMEWORK_PORT_LIBYUV_PORT_H_
#define MEDIAPIPE_FRAMEWORK_PORT_LIBYUV_PORT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "libyuv/convert.h"
#include "libyuv/convert_argb.h"
#include "libyuv/convert_from.h"

namespace mediapipe::libyuv_port {

struct PlaneRef {
  absl::Span<const uint8_t> data;
  int stride;
};

struct PlaneRefMut {
  absl::Span<uint8_t> data;
  int stride;
};

inline absl::Status ABGRToI420(PlaneRef src_abgr, PlaneRefMut dst_y,
                               PlaneRefMut dst_u, PlaneRefMut dst_v, int width,
                               int height) {
  int rv = libyuv::ABGRToI420(src_abgr.data.data(), src_abgr.stride,  //
                              dst_y.data.data(), dst_y.stride,        //
                              dst_u.data.data(), dst_u.stride,        //
                              dst_v.data.data(), dst_v.stride,        //
                              width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::ABGRToI420 failed");
}

inline absl::Status RAWToI420(PlaneRef src_raw, PlaneRefMut dst_y,
                              PlaneRefMut dst_u, PlaneRefMut dst_v, int width,
                              int height) {
  int rv = libyuv::RAWToI420(src_raw.data.data(), src_raw.stride,  //
                             dst_y.data.data(), dst_y.stride,      //
                             dst_u.data.data(), dst_u.stride,      //
                             dst_v.data.data(), dst_v.stride,      //
                             width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::RAWToI420 failed");
}

inline absl::Status I420ToNV12(PlaneRef src_y, PlaneRef src_u, PlaneRef src_v,
                               PlaneRefMut dst_y, PlaneRefMut dst_uv, int width,
                               int height) {
  int rv = libyuv::I420ToNV12(src_y.data.data(), src_y.stride,    //
                              src_u.data.data(), src_u.stride,    //
                              src_v.data.data(), src_v.stride,    //
                              dst_y.data.data(), dst_y.stride,    //
                              dst_uv.data.data(), dst_uv.stride,  //
                              width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::I420ToNV12 failed");
}

inline absl::Status H420ToRAW(PlaneRef src_y, PlaneRef src_u, PlaneRef src_v,
                              PlaneRefMut dst_raw, int width, int height) {
  int rv = libyuv::H420ToRAW(src_y.data.data(), src_y.stride,      //
                             src_u.data.data(), src_u.stride,      //
                             src_v.data.data(), src_v.stride,      //
                             dst_raw.data.data(), dst_raw.stride,  //
                             width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::H420ToRAW failed");
}

inline absl::Status I420ToRAW(PlaneRef src_y, PlaneRef src_u, PlaneRef src_v,
                              PlaneRefMut dst_raw, int width, int height) {
  int rv = libyuv::I420ToRAW(src_y.data.data(), src_y.stride,      //
                             src_u.data.data(), src_u.stride,      //
                             src_v.data.data(), src_v.stride,      //
                             dst_raw.data.data(), dst_raw.stride,  //
                             width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::I420ToRAW failed");
}

inline absl::Status NV12ToRAW(PlaneRef src_y, PlaneRef src_uv,
                              PlaneRefMut dst_raw, int width, int height) {
  int rv = libyuv::NV12ToRAW(src_y.data.data(), src_y.stride,      //
                             src_uv.data.data(), src_uv.stride,    //
                             dst_raw.data.data(), dst_raw.stride,  //
                             width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::NV12ToRAW failed");
}

inline absl::Status NV21ToRAW(PlaneRef src_y, PlaneRef src_vu,
                              PlaneRefMut dst_raw, int width, int height) {
  int rv = libyuv::NV21ToRAW(src_y.data.data(), src_y.stride,      //
                             src_vu.data.data(), src_vu.stride,    //
                             dst_raw.data.data(), dst_raw.stride,  //
                             width, height);
  return rv == 0 ? absl::OkStatus()
                 : absl::InternalError("libyuv::NV21ToRAW failed");
}

}  // namespace mediapipe::libyuv_port

#endif  // MEDIAPIPE_FRAMEWORK_PORT_LIBYUV_PORT_H_
