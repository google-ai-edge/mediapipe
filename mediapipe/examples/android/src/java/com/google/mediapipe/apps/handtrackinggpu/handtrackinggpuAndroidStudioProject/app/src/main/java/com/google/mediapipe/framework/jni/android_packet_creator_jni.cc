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

#include "mediapipe/java/com/google/mediapipe/framework/jni/android_packet_creator_jni.h"

#include <android/bitmap.h>

#include <cstring>
#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/colorspace.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"

namespace {

// Creates a new internal::PacketWithContext object, and returns the native
// handle.
int64_t CreatePacketWithContext(jlong context,
                                const mediapipe::Packet& packet) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  return mediapipe_graph->WrapPacketIntoContext(packet);
}

}  // namespace

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(
    nativeCreateRgbImageFrame)(JNIEnv* env, jobject thiz, jlong context,
                               jobject bitmap) {
  AndroidBitmapInfo info;
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOG(ERROR) << "AndroidBitmap_getInfo() failed with result code " << result;
    return 0L;
  }
  if (info.stride != info.width * 4) {
    LOG(ERROR) << "Bitmap stride: " << info.stride
               << "is not equal to 4 times bitmap width: " << info.width;
    return 0L;
  }
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, info.width, info.height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  void* pixel_addr = nullptr;
  result = AndroidBitmap_lockPixels(env, bitmap, &pixel_addr);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOG(ERROR) << "AndroidBitmap_lockPixels() failed with result code "
               << result;
    return 0L;
  }
  const uint8_t* rgba_data = static_cast<uint8_t*>(pixel_addr);
  mediapipe::android::RgbaToRgb(rgba_data, info.stride, info.width, info.height,
                                image_frame->MutablePixelData(),
                                image_frame->WidthStep());
  result = AndroidBitmap_unlockPixels(env, bitmap);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOG(ERROR) << "AndroidBitmap_unlockPixels() failed with result code "
               << result;
    return 0L;
  }
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(
    nativeCreateRgbaImageFrame)(JNIEnv* env, jobject thiz, jlong context,
                                jobject bitmap) {
  AndroidBitmapInfo info;
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOG(ERROR) << "AndroidBitmap_getInfo() failed with result code " << result;
    return 0L;
  }
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGBA, info.width, info.height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  int64_t buffer_size = info.stride * info.height;
  if (buffer_size != image_frame->PixelDataSize()) {
    LOG(ERROR) << "Bitmap stride: " << info.stride
               << " times bitmap height: " << info.height
               << " is not equal to the expected size: "
               << image_frame->PixelDataSize();
    return 0L;
  }
  void* pixel_addr = nullptr;
  result = AndroidBitmap_lockPixels(env, bitmap, &pixel_addr);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOG(ERROR) << "AndroidBitmap_lockPixels() failed with result code "
               << result;
    return 0L;
  }
  std::memcpy(image_frame->MutablePixelData(), pixel_addr,
              image_frame->PixelDataSize());
  result = AndroidBitmap_unlockPixels(env, bitmap);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOG(ERROR) << "AndroidBitmap_unlockPixels() failed with result code "
               << result;
    return 0L;
  }
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}
