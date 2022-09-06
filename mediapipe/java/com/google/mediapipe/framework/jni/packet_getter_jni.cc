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

#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_getter_jni.h"

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/colorspace.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
using mediapipe::android::SerializedMessageIds;
using mediapipe::android::ThrowIfError;

template <typename T>
const T& GetFromNativeHandle(int64_t packet_handle) {
  return mediapipe::android::Graph::GetPacketFromHandle(packet_handle).Get<T>();
}
}  // namespace

JNIEXPORT jlong JNICALL PACKET_GETTER_METHOD(nativeGetPacketFromReference)(
    JNIEnv* env, jobject thiz, jlong packet) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet)
          .Get<std::unique_ptr<mediapipe::SyncedPacket>>()
          ->Get();
  auto mediapipe_graph =
      mediapipe::android::Graph::GetContextFromHandle(packet);
  return mediapipe_graph->WrapPacketIntoContext(mediapipe_packet);
}

JNIEXPORT jlongArray JNICALL PACKET_GETTER_METHOD(nativeGetPairPackets)(
    JNIEnv* env, jobject thiz, jlong packet) {
  jlongArray return_handles = env->NewLongArray(2);
  auto pair_packets =
      GetFromNativeHandle<std::pair<mediapipe::Packet, mediapipe::Packet>>(
          packet);
  auto mediapipe_graph =
      mediapipe::android::Graph::GetContextFromHandle(packet);
  int64_t handles[2];
  handles[0] = mediapipe_graph->WrapPacketIntoContext(pair_packets.first);
  handles[1] = mediapipe_graph->WrapPacketIntoContext(pair_packets.second);
  env->SetLongArrayRegion(return_handles, 0, 2,
                          reinterpret_cast<const jlong*>(handles));
  return return_handles;
}

JNIEXPORT jlongArray JNICALL PACKET_GETTER_METHOD(nativeGetVectorPackets)(
    JNIEnv* env, jobject thiz, jlong packet) {
  auto vector_packets =
      GetFromNativeHandle<std::vector<mediapipe::Packet>>(packet);
  auto mediapipe_graph =
      mediapipe::android::Graph::GetContextFromHandle(packet);
  jlongArray return_handles = env->NewLongArray(vector_packets.size());
  std::vector<int64_t> handles(vector_packets.size());
  for (int i = 0; i < vector_packets.size(); ++i) {
    handles[i] = mediapipe_graph->WrapPacketIntoContext(vector_packets[i]);
  }
  env->SetLongArrayRegion(return_handles, 0, handles.size(),
                          reinterpret_cast<const jlong*>(&(handles[0])));
  return return_handles;
}

JNIEXPORT jshort JNICALL PACKET_GETTER_METHOD(nativeGetInt16)(JNIEnv* env,
                                                              jobject thiz,
                                                              jlong packet) {
  return GetFromNativeHandle<int16_t>(packet);
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetInt32)(JNIEnv* env,
                                                            jobject thiz,
                                                            jlong packet) {
  return GetFromNativeHandle<int32_t>(packet);
}

JNIEXPORT jlong JNICALL PACKET_GETTER_METHOD(nativeGetInt64)(JNIEnv* env,
                                                             jobject thiz,
                                                             jlong packet) {
  return GetFromNativeHandle<int64_t>(packet);
}

JNIEXPORT jfloat JNICALL PACKET_GETTER_METHOD(nativeGetFloat32)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong packet) {
  return GetFromNativeHandle<float>(packet);
}

JNIEXPORT jdouble JNICALL PACKET_GETTER_METHOD(nativeGetFloat64)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet) {
  return GetFromNativeHandle<double>(packet);
}

JNIEXPORT jboolean JNICALL PACKET_GETTER_METHOD(nativeGetBool)(JNIEnv* env,
                                                               jobject thiz,
                                                               jlong packet) {
  return GetFromNativeHandle<bool>(packet);
}

JNIEXPORT jstring JNICALL PACKET_GETTER_METHOD(nativeGetString)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong packet) {
  const std::string& value = GetFromNativeHandle<std::string>(packet);
  return env->NewStringUTF(value.c_str());
}

JNIEXPORT jbyteArray JNICALL
PACKET_GETTER_METHOD(nativeGetBytes)(JNIEnv* env, jobject thiz, jlong packet) {
  const std::string& value = GetFromNativeHandle<std::string>(packet);
  jbyteArray data = env->NewByteArray(value.length());
  env->SetByteArrayRegion(data, 0, value.length(),
                          reinterpret_cast<const jbyte*>(value.c_str()));
  return data;
}

JNIEXPORT jbyteArray JNICALL PACKET_GETTER_METHOD(nativeGetProtoBytes)(
    JNIEnv* env, jobject thiz, jlong packet) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  const auto& proto_message = mediapipe_packet.GetProtoMessageLite();
  std::string serialized;
  proto_message.SerializeToString(&serialized);
  jbyteArray data = env->NewByteArray(serialized.size());
  env->SetByteArrayRegion(data, 0, serialized.size(),
                          reinterpret_cast<const jbyte*>(serialized.c_str()));
  return data;
}

JNIEXPORT void JNICALL PACKET_GETTER_METHOD(nativeGetProto)(JNIEnv* env,
                                                            jobject thiz,
                                                            jlong packet,
                                                            jobject result) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  absl::Status status = mediapipe_packet.ValidateAsProtoMessageLite();
  if (!ThrowIfError(env, status)) {
    // Convert type_name and value to Java data.
    const auto& proto_message = mediapipe_packet.GetProtoMessageLite();
    std::string type_name = proto_message.GetTypeName();
    jstring j_type_name = env->NewStringUTF(type_name.c_str());
    std::string proto_bytes;
    proto_message.SerializeToString(&proto_bytes);
    jbyteArray j_proto_bytes = env->NewByteArray(proto_bytes.length());
    env->SetByteArrayRegion(
        j_proto_bytes, 0, proto_bytes.length(),
        reinterpret_cast<const jbyte*>(proto_bytes.c_str()));

    // Set type_name and value in the result Java object.
    static SerializedMessageIds ids(env, result);
    env->SetObjectField(result, ids.type_name_id, j_type_name);
    env->SetObjectField(result, ids.value_id, j_proto_bytes);
  }
}

JNIEXPORT jobjectArray JNICALL PACKET_GETTER_METHOD(nativeGetProtoVector)(
    JNIEnv* env, jobject thiz, jlong packet) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  auto get_proto_vector = mediapipe_packet.GetVectorOfProtoMessageLitePtrs();
  if (!get_proto_vector.ok()) {
    env->Throw(mediapipe::android::CreateMediaPipeException(
        env, get_proto_vector.status()));
  }
  const std::vector<const ::mediapipe::proto_ns::MessageLite*>& proto_vector =
      get_proto_vector.value();
  // TODO: move to register natives.
  jclass byte_array_cls = env->FindClass("[B");
  jobjectArray proto_array =
      env->NewObjectArray(proto_vector.size(), byte_array_cls, nullptr);
  env->DeleteLocalRef(byte_array_cls);
  for (int i = 0; i < proto_vector.size(); ++i) {
    const ::mediapipe::proto_ns::MessageLite* proto_message = proto_vector[i];

    // Convert the proto object into a Java byte array.
    std::string serialized;
    proto_message->SerializeToString(&serialized);
    jbyteArray byte_array = env->NewByteArray(serialized.size());
    env->SetByteArrayRegion(byte_array, 0, serialized.size(),
                            reinterpret_cast<const jbyte*>(serialized.c_str()));

    // Add the serialized proto byte_array to the output array.
    env->SetObjectArrayElement(proto_array, i, byte_array);
    env->DeleteLocalRef(byte_array);
  }

  return proto_array;
}

JNIEXPORT jshortArray JNICALL PACKET_GETTER_METHOD(nativeGetInt16Vector)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const std::vector<int16_t>& values =
      GetFromNativeHandle<std::vector<int16_t>>(packet);
  jshortArray result = env->NewShortArray(values.size());
  env->SetShortArrayRegion(result, 0, values.size(), &(values[0]));
  return result;
}

JNIEXPORT jintArray JNICALL PACKET_GETTER_METHOD(nativeGetInt32Vector)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const std::vector<int>& values =
      GetFromNativeHandle<std::vector<int>>(packet);
  jintArray result = env->NewIntArray(values.size());
  env->SetIntArrayRegion(result, 0, values.size(), &(values[0]));
  return result;
}

JNIEXPORT jlongArray JNICALL PACKET_GETTER_METHOD(nativeGetInt64Vector)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const std::vector<int64_t>& values =
      GetFromNativeHandle<std::vector<int64_t>>(packet);
  jlongArray result = env->NewLongArray(values.size());
  // 64 bit builds treat jlong as long long, and int64_t as long int, although
  // both are 64 bits, but still need to use the reinterpret_cast to avoid the
  // compiling error.
  env->SetLongArrayRegion(result, 0, values.size(),
                          reinterpret_cast<const jlong*>(&(values[0])));
  return result;
}

JNIEXPORT jfloatArray JNICALL PACKET_GETTER_METHOD(nativeGetFloat32Vector)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const std::vector<float>& values =
      GetFromNativeHandle<std::vector<float>>(packet);
  jfloatArray result = env->NewFloatArray(values.size());
  env->SetFloatArrayRegion(result, 0, values.size(), &(values[0]));
  return result;
}

JNIEXPORT jdoubleArray JNICALL PACKET_GETTER_METHOD(nativeGetFloat64Vector)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const std::vector<double>& values =
      GetFromNativeHandle<std::vector<double>>(packet);
  jdoubleArray result = env->NewDoubleArray(values.size());
  env->SetDoubleArrayRegion(result, 0, values.size(), &(values[0]));
  return result;
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetImageWidth)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  const bool is_image =
      mediapipe_packet.ValidateAsType<mediapipe::Image>().ok();
  const mediapipe::ImageFrame& image =
      is_image ? *GetFromNativeHandle<mediapipe::Image>(packet)
                      .GetImageFrameSharedPtr()
                      .get()
               : GetFromNativeHandle<mediapipe::ImageFrame>(packet);
  return image.Width();
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetImageHeight)(
    JNIEnv* env, jobject thiz, jlong packet) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  const bool is_image =
      mediapipe_packet.ValidateAsType<mediapipe::Image>().ok();
  const mediapipe::ImageFrame& image =
      is_image ? *GetFromNativeHandle<mediapipe::Image>(packet)
                      .GetImageFrameSharedPtr()
                      .get()
               : GetFromNativeHandle<mediapipe::ImageFrame>(packet);
  return image.Height();
}

JNIEXPORT jboolean JNICALL PACKET_GETTER_METHOD(nativeGetImageData)(
    JNIEnv* env, jobject thiz, jlong packet, jobject byte_buffer) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  const bool is_image =
      mediapipe_packet.ValidateAsType<mediapipe::Image>().ok();
  const mediapipe::ImageFrame& image =
      is_image ? *GetFromNativeHandle<mediapipe::Image>(packet)
                      .GetImageFrameSharedPtr()
                      .get()
               : GetFromNativeHandle<mediapipe::ImageFrame>(packet);

  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);

  // Assume byte buffer stores pixel data contiguously.
  const int expected_buffer_size = image.Width() * image.Height() *
                                   image.ByteDepth() * image.NumberOfChannels();
  if (buffer_size != expected_buffer_size) {
    LOG(ERROR) << "Expected buffer size " << expected_buffer_size
               << " got: " << buffer_size << ", width " << image.Width()
               << ", height " << image.Height() << ", channels "
               << image.NumberOfChannels();
    return false;
  }

  switch (image.ByteDepth()) {
    case 1: {
      uint8* data =
          static_cast<uint8*>(env->GetDirectBufferAddress(byte_buffer));
      image.CopyToBuffer(data, expected_buffer_size);
      break;
    }
    case 2: {
      uint16* data =
          static_cast<uint16*>(env->GetDirectBufferAddress(byte_buffer));
      image.CopyToBuffer(data, expected_buffer_size);
      break;
    }
    case 4: {
      float* data =
          static_cast<float*>(env->GetDirectBufferAddress(byte_buffer));
      image.CopyToBuffer(data, expected_buffer_size);
      break;
    }
    default: {
      return false;
    }
  }
  return true;
}

JNIEXPORT jboolean JNICALL PACKET_GETTER_METHOD(nativeGetRgbaFromRgb)(
    JNIEnv* env, jobject thiz, jlong packet, jobject byte_buffer) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  const bool is_image =
      mediapipe_packet.ValidateAsType<mediapipe::Image>().ok();
  const mediapipe::ImageFrame& image =
      is_image ? *GetFromNativeHandle<mediapipe::Image>(packet)
                      .GetImageFrameSharedPtr()
                      .get()
               : GetFromNativeHandle<mediapipe::ImageFrame>(packet);
  uint8_t* rgba_data =
      static_cast<uint8_t*>(env->GetDirectBufferAddress(byte_buffer));
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (buffer_size != image.Width() * image.Height() * 4) {
    LOG(ERROR) << "Buffer size has to be width*height*4\n"
               << "Image width: " << image.Width()
               << ", Image height: " << image.Height()
               << ", Buffer size: " << buffer_size << ", Buffer size needed: "
               << image.Width() * image.Height() * 4;
    return false;
  }
  mediapipe::android::RgbToRgba(image.PixelData(), image.WidthStep(),
                                image.Width(), image.Height(), rgba_data,
                                image.Width() * 4, 255);
  return true;
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetVideoHeaderWidth)(
    JNIEnv* env, jobject thiz, jlong packet) {
  return GetFromNativeHandle<mediapipe::VideoHeader>(packet).width;
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetVideoHeaderHeight)(
    JNIEnv* env, jobject thiz, jlong packet) {
  return GetFromNativeHandle<mediapipe::VideoHeader>(packet).height;
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(
    nativeGetTimeSeriesHeaderNumChannels)(JNIEnv* env, jobject thiz,
                                          jlong packet) {
  return GetFromNativeHandle<mediapipe::TimeSeriesHeader>(packet)
      .num_channels();
}

JNIEXPORT jdouble JNICALL PACKET_GETTER_METHOD(
    nativeGetTimeSeriesHeaderSampleRate)(JNIEnv* env, jobject thiz,
                                         jlong packet) {
  return GetFromNativeHandle<mediapipe::TimeSeriesHeader>(packet).sample_rate();
}

JNIEXPORT jbyteArray JNICALL PACKET_GETTER_METHOD(nativeGetAudioData)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const mediapipe::Matrix& audio_mat =
      GetFromNativeHandle<mediapipe::Matrix>(packet);
  int num_channels = audio_mat.rows();
  int num_samples = audio_mat.cols();
  int data_size = num_channels * num_samples * 2;
  const int kMultiplier = 1 << 15;
  jbyteArray byte_data = env->NewByteArray(data_size);
  int offset = 0;
  for (int sample = 0; sample < num_samples; ++sample) {
    for (int channel = 0; channel < num_channels; ++channel) {
      int16 value =
          static_cast<int16>(audio_mat(channel, sample) * kMultiplier);
      // The java and native has the same byte order, by default is little
      // Endian, we can safely copy data directly, we have tests to cover this.
      env->SetByteArrayRegion(byte_data, offset, 2,
                              reinterpret_cast<const jbyte*>(&value));
      offset += 2;
    }
  }
  return byte_data;
}

JNIEXPORT jfloatArray JNICALL PACKET_GETTER_METHOD(nativeGetMatrixData)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const mediapipe::Matrix& audio_mat =
      GetFromNativeHandle<mediapipe::Matrix>(packet);
  int rows = audio_mat.rows();
  int cols = audio_mat.cols();
  jfloatArray float_data = env->NewFloatArray(rows * cols);
  env->SetFloatArrayRegion(float_data, 0, rows * cols,
                           reinterpret_cast<const jfloat*>(audio_mat.data()));
  return float_data;
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetMatrixRows)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet) {
  return GetFromNativeHandle<mediapipe::Matrix>(packet).rows();
}

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetMatrixCols)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet) {
  return GetFromNativeHandle<mediapipe::Matrix>(packet).cols();
}

#if !MEDIAPIPE_DISABLE_GPU

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetGpuBufferName)(
    JNIEnv* env, jobject thiz, jlong packet) {
  const mediapipe::GpuBuffer& gpu_buffer =
      GetFromNativeHandle<mediapipe::GpuBuffer>(packet);
  // gpu_buffer.name() returns a GLuint. Make sure the cast to jint is safe.
  static_assert(sizeof(GLuint) <= sizeof(jint),
                "The cast to jint may truncate GLuint");
  return static_cast<jint>(
      gpu_buffer.internal_storage<mediapipe::GlTextureBuffer>()->name());
}

JNIEXPORT jlong JNICALL PACKET_GETTER_METHOD(nativeGetGpuBuffer)(
    JNIEnv* env, jobject thiz, jlong packet, jboolean wait_on_cpu) {
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  mediapipe::GlTextureBufferSharedPtr ptr;
  if (mediapipe_packet.ValidateAsType<mediapipe::Image>().ok()) {
    auto mediapipe_graph =
        mediapipe::android::Graph::GetContextFromHandle(packet);
    auto gl_context = mediapipe_graph->GetGpuResources()->gl_context();
    auto status =
        gl_context->Run([gl_context, mediapipe_packet, &ptr]() -> absl::Status {
          const mediapipe::Image& buffer =
              mediapipe_packet.Get<mediapipe::Image>();
          ptr = buffer.GetGlTextureBufferSharedPtr();
          return absl::OkStatus();
        });
  } else {
    const mediapipe::GpuBuffer& buffer =
        mediapipe_packet.Get<mediapipe::GpuBuffer>();
    ptr = buffer.internal_storage<mediapipe::GlTextureBuffer>();
  }
  if (wait_on_cpu) {
    ptr->WaitUntilComplete();
  }
  return reinterpret_cast<intptr_t>(
      new mediapipe::GlTextureBufferSharedPtr(ptr));
}

#endif  // !MEDIAPIPE_DISABLE_GPU
