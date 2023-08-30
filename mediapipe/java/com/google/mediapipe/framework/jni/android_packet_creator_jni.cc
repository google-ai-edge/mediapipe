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

#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
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

// Create 3 or 4 channel 8-bit ImageFrame shared pointer from a Java Bitmap.
std::unique_ptr<mediapipe::ImageFrame> CreateImageFrameFromBitmap(
    JNIEnv* env, jobject bitmap, int width, int height, int stride,
    mediapipe::ImageFormat::Format format) {
  auto image_frame = std::make_unique<mediapipe::ImageFrame>(
      format, width, height,
      mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);

  void* pixel_addr = nullptr;
  int result = AndroidBitmap_lockPixels(env, bitmap, &pixel_addr);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    ABSL_LOG(ERROR) << "AndroidBitmap_lockPixels() failed with result code "
                    << result;
    return nullptr;
  }

  if (format == mediapipe::ImageFormat::SRGBA) {
    const int64_t buffer_size = stride * height;
    if (buffer_size != image_frame->PixelDataSize()) {
      ABSL_LOG(ERROR) << "Bitmap stride: " << stride
                      << " times bitmap height: " << height
                      << " is not equal to the expected size: "
                      << image_frame->PixelDataSize();
      return nullptr;
    }
    std::memcpy(image_frame->MutablePixelData(), pixel_addr,
                image_frame->PixelDataSize());
  } else if (format == mediapipe::ImageFormat::SRGB) {
    if (stride != width * 4) {
      ABSL_LOG(ERROR) << "Bitmap stride: " << stride
                      << "is not equal to 4 times bitmap width: " << width;
      return nullptr;
    }
    const uint8_t* rgba_data = static_cast<uint8_t*>(pixel_addr);
    mediapipe::android::RgbaToRgb(rgba_data, stride, width, height,
                                  image_frame->MutablePixelData(),
                                  image_frame->WidthStep());
  } else {
    ABSL_LOG(ERROR) << "unsupported image format: " << format;
    return nullptr;
  }

  result = AndroidBitmap_unlockPixels(env, bitmap);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    ABSL_LOG(ERROR) << "AndroidBitmap_unlockPixels() failed with result code "
                    << result;
    return nullptr;
  }

  return image_frame;
}

}  // namespace

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(
    nativeCreateRgbImageFrame)(JNIEnv* env, jobject thiz, jlong context,
                               jobject bitmap) {
  AndroidBitmapInfo info;
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    ABSL_LOG(ERROR) << "AndroidBitmap_getInfo() failed with result code "
                    << result;
    return 0L;
  }

  auto image_frame =
      CreateImageFrameFromBitmap(env, bitmap, info.width, info.height,
                                 info.stride, mediapipe::ImageFormat::SRGB);
  if (nullptr == image_frame) return 0L;

  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(
    nativeCreateRgbaImageFrame)(JNIEnv* env, jobject thiz, jlong context,
                                jobject bitmap) {
  AndroidBitmapInfo info;
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    ABSL_LOG(ERROR) << "AndroidBitmap_getInfo() failed with result code "
                    << result;
    return 0L;
  }

  auto image_frame =
      CreateImageFrameFromBitmap(env, bitmap, info.width, info.height,
                                 info.stride, mediapipe::ImageFormat::SRGBA);
  if (nullptr == image_frame) return 0L;

  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(nativeCreateRgbaImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject bitmap) {
  AndroidBitmapInfo info;
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    ABSL_LOG(ERROR) << "AndroidBitmap_getInfo() failed with result code "
                    << result;
    return 0L;
  }

  auto image_frame =
      CreateImageFrameFromBitmap(env, bitmap, info.width, info.height,
                                 info.stride, mediapipe::ImageFormat::SRGBA);
  if (nullptr == image_frame) return 0L;

  mediapipe::Packet packet =
      mediapipe::MakePacket<mediapipe::Image>(std::move(image_frame));
  return CreatePacketWithContext(context, packet);
}
