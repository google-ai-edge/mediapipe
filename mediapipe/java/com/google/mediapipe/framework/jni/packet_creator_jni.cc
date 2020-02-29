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

#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_creator_jni.h"

#include <cstring>
#include <memory>

#include "mediapipe/framework/camera_intrinsics.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/colorspace.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#ifndef MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GPU)

namespace {

template <class T>
int64_t CreatePacketScalar(jlong context, const T& value) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  mediapipe::Packet packet = mediapipe::Adopt(new T(value));
  return mediapipe_graph->WrapPacketIntoContext(packet);
}

// Creates a new internal::PacketWithContext object, and returns the native
// handle.
int64_t CreatePacketWithContext(jlong context,
                                const mediapipe::Packet& packet) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  return mediapipe_graph->WrapPacketIntoContext(packet);
}

}  // namespace

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateReferencePacket)(
    JNIEnv* env, jobject thiz, jlong context, jlong packet) {
  auto mediapipe_graph = reinterpret_cast<mediapipe::android::Graph*>(context);
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  auto reference_packet = mediapipe::AdoptAsUniquePtr(
      new mediapipe::SyncedPacket(mediapipe_packet));
  // assigned the initial value of the packet reference.
  return mediapipe_graph->WrapPacketIntoContext(reference_packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  const void* data = env->GetDirectBufferAddress(byte_buffer);
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, width, height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (buffer_size != image_frame->PixelDataSize()) {
    LOG(ERROR) << "The input image buffer should have 4 bytes alignment.";
    LOG(ERROR) << "Buffer size: " << buffer_size
               << ", Buffer size needed: " << image_frame->PixelDataSize()
               << ", Image width: " << width;
    return 0L;
  }
  std::memcpy(image_frame->MutablePixelData(), data,
              image_frame->PixelDataSize());
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbImageFromRgba)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  const uint8_t* rgba_data =
      static_cast<uint8_t*>(env->GetDirectBufferAddress(byte_buffer));
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, width, height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (buffer_size != width * height * 4) {
    LOG(ERROR) << "Please check the input buffer size.";
    LOG(ERROR) << "Buffer size: " << buffer_size
               << ", Buffer size needed: " << width * height * 4
               << ", Image width: " << width;
    return 0L;
  }
  mediapipe::android::RgbaToRgb(rgba_data, width * 4, width, height,
                                image_frame->MutablePixelData(),
                                image_frame->WidthStep());
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGrayscaleImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::GRAY8, width, height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (buffer_size != width * height) {
    LOG(ERROR) << "Please check the input buffer size.";
    LOG(ERROR) << "Buffer size: " << buffer_size
               << ", Buffer size needed: " << width * height
               << ", Image height: " << height;
    return 0L;
  }

  int width_step = image_frame->WidthStep();
  // Copy buffer data to image frame's pixel_data_.
  const char* src_row =
      reinterpret_cast<const char*>(env->GetDirectBufferAddress(byte_buffer));
  char* dst_row = reinterpret_cast<char*>(image_frame->MutablePixelData());
  for (int i = height; i > 0; --i) {
    std::memcpy(dst_row, src_row, width);
    src_row += width;
    dst_row += width_step;
  }
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloatImageFrame)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  const void* data = env->GetDirectBufferAddress(byte_buffer);
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::VEC32F1, width, height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (buffer_size != image_frame->PixelDataSize()) {
    LOG(ERROR) << "Please check the input buffer size.";
    LOG(ERROR) << "Buffer size: " << buffer_size
               << ", Buffer size needed: " << image_frame->PixelDataSize()
               << ", Image width: " << width;
    return 0L;
  }
  std::memcpy(image_frame->MutablePixelData(), data,
              image_frame->PixelDataSize());
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbaImageFrame)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  const void* rgba_data = env->GetDirectBufferAddress(byte_buffer);
  auto image_frame = absl::make_unique<::mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGBA, width, height,
      ::mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (buffer_size != image_frame->PixelDataSize()) {
    LOG(ERROR) << "Please check the input buffer size.";
    LOG(ERROR) << "Buffer size: " << buffer_size
               << ", Buffer size needed: " << image_frame->PixelDataSize()
               << ", Image width: " << width;
    return 0L;
  }
  std::memcpy(image_frame->MutablePixelData(), rgba_data,
              image_frame->PixelDataSize());
  mediapipe::Packet packet = mediapipe::Adopt(image_frame.release());
  return CreatePacketWithContext(context, packet);
}

static mediapipe::Packet createAudioPacket(const uint8_t* audio_sample,
                                           int num_samples, int num_channels) {
  std::unique_ptr<::mediapipe::Matrix> matrix(
      new ::mediapipe::Matrix(num_channels, num_samples));
  // Preparing and normalize the audio data.
  // kMultiplier is same as what used in av_sync_media_decoder.cc.
  static const float kMultiplier = 1.f / (1 << 15);
  // We try to not assume the Endian order of the data.
  for (int sample = 0; sample < num_samples; ++sample) {
    for (int channel = 0; channel < num_channels; ++channel) {
      int16_t value = (audio_sample[1] & 0xff) << 8 | audio_sample[0];
      (*matrix)(channel, sample) = kMultiplier * value;
      audio_sample += 2;
    }
  }
  return mediapipe::Adopt(matrix.release());
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateAudioPacket)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data, jint offset,
    jint num_channels, jint num_samples) {
  // Note, audio_data_ref is really a const jbyte* but this clashes with the
  // the expectation of ReleaseByteArrayElements below.
  jbyte* audio_data_ref = env->GetByteArrayElements(data, nullptr);
  const uint8_t* audio_sample =
      reinterpret_cast<uint8_t*>(audio_data_ref) + offset;
  mediapipe::Packet packet =
      createAudioPacket(audio_sample, num_samples, num_channels);
  env->ReleaseByteArrayElements(data, audio_data_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateAudioPacketDirect)(
    JNIEnv* env, jobject thiz, jlong context, jobject data, jint num_channels,
    jint num_samples) {
  const uint8_t* audio_sample =
      reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(data));
  mediapipe::Packet packet =
      createAudioPacket(audio_sample, num_samples, num_channels);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt16)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jshort value) {
  return CreatePacketScalar<int16_t>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jint value) {
  return CreatePacketScalar<int>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt64)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jlong value) {
  return CreatePacketScalar<int64_t>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32)(
    JNIEnv* env, jobject thiz, jlong context, jfloat value) {
  return CreatePacketScalar<float>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat64)(
    JNIEnv* env, jobject thiz, jlong context, jdouble value) {
  return CreatePacketScalar<double>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateBool)(
    JNIEnv* env, jobject thiz, jlong context, jboolean value) {
  return CreatePacketScalar<bool>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateString)(
    JNIEnv* env, jobject thiz, jlong context, jstring value) {
  return CreatePacketScalar<std::string>(
      context, mediapipe::android::JStringToStdString(env, value));
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateVideoHeader)(
    JNIEnv* env, jobject thiz, jlong context, jint width, jint height) {
  mediapipe::VideoHeader header;
  header.format = mediapipe::ImageFormat::SRGB;
  header.width = width;
  header.height = height;
  return CreatePacketScalar<mediapipe::VideoHeader>(context, header);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateTimeSeriesHeader)(
    JNIEnv* env, jobject thiz, jlong context, jint num_channels,
    jdouble sample_rate) {
  mediapipe::TimeSeriesHeader header;
  header.set_num_channels(num_channels);
  header.set_sample_rate(sample_rate);
  return CreatePacketScalar<mediapipe::TimeSeriesHeader>(context, header);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateMatrix)(
    JNIEnv* env, jobject thiz, jlong context, jint rows, jint cols,
    jfloatArray data) {
  if (env->GetArrayLength(data) != rows * cols) {
    LOG(ERROR) << "Please check the matrix data size, "
                  "has to be rows * cols = "
               << rows * cols;
    return 0L;
  }
  std::unique_ptr<::mediapipe::Matrix> matrix(
      new ::mediapipe::Matrix(rows, cols));
  // The java and native has the same byte order, by default is little Endian,
  // we can safely copy data directly, we have tests to cover this.
  env->GetFloatArrayRegion(data, 0, rows * cols, matrix->data());
  mediapipe::Packet packet = mediapipe::Adopt(matrix.release());
  return CreatePacketWithContext(context, packet);
}

#ifndef MEDIAPIPE_DISABLE_GPU

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGpuBuffer)(
    JNIEnv* env, jobject thiz, jlong context, jint name, jint width,
    jint height, jobject texture_release_callback) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  auto* gpu_resources = mediapipe_graph->GetGpuResources();
  CHECK(gpu_resources) << "Cannot create a mediapipe::GpuBuffer packet on a "
                          "graph without GPU support";
  mediapipe::GlTextureBuffer::DeletionCallback cc_callback;

  if (texture_release_callback) {
    // TODO: see if this can be cached.
    // Note: we don't get this from the object because people may pass a
    // subclass of PacketCreator, and the method is private.
    jclass my_class =
        env->FindClass("com/google/mediapipe/framework/PacketCreator");
    jmethodID release_method =
        env->GetMethodID(my_class, "releaseWithSyncToken",
                         "(JL"
                         "com/google/mediapipe/framework/TextureReleaseCallback"
                         ";)V");
    CHECK(release_method);
    env->DeleteLocalRef(my_class);

    jobject java_callback = env->NewGlobalRef(texture_release_callback);
    jobject packet_creator = env->NewGlobalRef(thiz);
    cc_callback = [mediapipe_graph, packet_creator, release_method,
                   java_callback](mediapipe::GlSyncToken release_token) {
      JNIEnv* env = mediapipe::java::GetJNIEnv();

      jlong raw_token = reinterpret_cast<jlong>(
          new mediapipe::GlSyncToken(std::move(release_token)));
      env->CallVoidMethod(packet_creator, release_method, raw_token,
                          java_callback);

      // Note that this callback is called only once, and is not saved
      // anywhere else, so we can and should delete it here.
      env->DeleteGlobalRef(java_callback);
      env->DeleteGlobalRef(packet_creator);
    };
  }
  mediapipe::Packet packet = mediapipe::MakePacket<mediapipe::GpuBuffer>(
      mediapipe::GlTextureBuffer::Wrap(GL_TEXTURE_2D, name, width, height,
                                       mediapipe::GpuBufferFormat::kBGRA32,
                                       cc_callback));
  return CreatePacketWithContext(context, packet);
}

#endif  // !defined(MEDIAPIPE_DISABLE_GPU)

// TODO: Add vector creators.

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32Array)(
    JNIEnv* env, jobject thiz, jlong context, jfloatArray data) {
  jsize count = env->GetArrayLength(data);
  jfloat* data_ref = env->GetFloatArrayElements(data, nullptr);
  float* floats = new float[count];
  // jfloat is a "machine-dependent native type" which represents a 32-bit
  // float. C++ makes no guarantees about the size of floating point types, and
  // some exotic architectures don't even have 32-bit floats (or even binary
  // floats), but on all architectures we care about this is a float.
  static_assert(std::is_same<float, jfloat>::value, "jfloat must be float");
  std::memcpy(floats, data_ref, count * sizeof(float));
  env->ReleaseFloatArrayElements(data, data_ref, JNI_ABORT);

  // The reinterpret_cast is needed to make the Adopt template recognize
  // that this is an array - this way Holder will call delete[].
  mediapipe::Packet packet =
      mediapipe::Adopt(reinterpret_cast<float(*)[]>(floats));
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32Array)(
    JNIEnv* env, jobject thiz, jlong context, jintArray data) {
  jsize count = env->GetArrayLength(data);
  jint* data_ref = env->GetIntArrayElements(data, nullptr);
  int32_t* ints = new int32_t[count];
  static_assert(std::is_same<int32_t, jint>::value, "jint must be int32_t");
  std::memcpy(ints, data_ref, count * sizeof(int32_t));
  env->ReleaseIntArrayElements(data, data_ref, JNI_ABORT);

  // The reinterpret_cast is needed to make the Adopt template recognize
  // that this is an array - this way Holder will call delete[].
  mediapipe::Packet packet =
      mediapipe::Adopt(reinterpret_cast<int32_t(*)[]>(ints));
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateStringFromByteArray)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data) {
  jsize count = env->GetArrayLength(data);
  jbyte* data_ref = env->GetByteArrayElements(data, nullptr);
  mediapipe::Packet packet = mediapipe::Adopt(
      new std::string(reinterpret_cast<char*>(data_ref), count));
  env->ReleaseByteArrayElements(data, data_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCalculatorOptions)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data) {
  jsize count = env->GetArrayLength(data);
  jbyte* data_ref = env->GetByteArrayElements(data, nullptr);
  auto options = absl::make_unique<mediapipe::CalculatorOptions>();
  if (!options->ParseFromArray(data_ref, count)) {
    LOG(ERROR) << "Parsing binary-encoded CalculatorOptions failed.";
    return 0L;
  }
  mediapipe::Packet packet = mediapipe::Adopt(options.release());
  env->ReleaseByteArrayElements(data, data_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCameraIntrinsics)(
    JNIEnv* env, jobject thiz, jlong context, jfloat fx, jfloat fy, jfloat cx,
    jfloat cy, jfloat width, jfloat height) {
  mediapipe::Packet packet =
      mediapipe::MakePacket<CameraIntrinsics>(fx, fy, cx, cy, width, height);
  return CreatePacketWithContext(context, packet);
}
