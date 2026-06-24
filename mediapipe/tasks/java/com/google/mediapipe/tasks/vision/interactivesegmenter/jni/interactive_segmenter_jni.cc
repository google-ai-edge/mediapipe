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

#include <jni.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/host_environment.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter/proto/stroke.pb.h"

namespace {

using ::mediapipe::android::JStringToStdString;
using ::mediapipe::android::ThrowIfError;
using ::mediapipe::tasks::core::ConvertProtoToBaseOptions;
using ::mediapipe::tasks::core::ToHostEnvironment;
using ::mediapipe::tasks::core::ToHostSystem;
using ::mediapipe::tasks::vision::interactive_segmenter::InteractiveSegmenter;
using ::mediapipe::tasks::vision::interactive_segmenter::
    InteractiveSegmenterOptions;
}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_google_mediapipe_tasks_vision_interactivesegmenter_InteractiveSegmenter_nativeCreate(
    JNIEnv* env, jclass clazz, jbyteArray base_options_bytes, jstring app_id,
    jstring app_version, jint host_environment, jint host_system) {
  jsize base_options_len = env->GetArrayLength(base_options_bytes);
  jbyte* base_options_data =
      env->GetByteArrayElements(base_options_bytes, nullptr);

  mediapipe::tasks::core::proto::BaseOptions base_options_proto;
  bool base_options_parsed =
      base_options_proto.ParseFromArray(base_options_data, base_options_len);
  env->ReleaseByteArrayElements(base_options_bytes, base_options_data,
                                JNI_ABORT);
  if (!base_options_parsed) {
    ThrowIfError(
        env, absl::InvalidArgumentError("Failed to parse BaseOptions proto."));
    return 0;
  }

  std::string app_id_str =
      app_id != nullptr ? JStringToStdString(env, app_id) : "";
  std::string app_version_str =
      app_version != nullptr ? JStringToStdString(env, app_version) : "";

  auto options = std::make_unique<InteractiveSegmenterOptions>();
  options->base_options =
      ConvertProtoToBaseOptions(std::move(base_options_proto));

  options->base_options.host_environment = ToHostEnvironment(host_environment);
  options->base_options.host_system = ToHostSystem(host_system);
  options->base_options.app_id = app_id_str;
  options->base_options.app_version = app_version_str;

  absl::StatusOr<std::unique_ptr<InteractiveSegmenter>> segmenter_or =
      InteractiveSegmenter::Create(std::move(options));
  if (!segmenter_or.ok()) {
    ThrowIfError(env, segmenter_or.status());
    return 0;
  }
  return reinterpret_cast<jlong>(segmenter_or.value().release());
}

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_vision_interactivesegmenter_InteractiveSegmenter_nativeSetImage(
    JNIEnv* env, jobject thiz, jlong segmenter_handle,
    jlong image_packet_handle) {
  InteractiveSegmenter* segmenter =
      reinterpret_cast<InteractiveSegmenter*>(segmenter_handle);

  mediapipe::Packet packet =
      mediapipe::android::Graph::GetPacketFromHandle(image_packet_handle);
  if (!packet.ValidateAsType<mediapipe::Image>().ok()) {
    ThrowIfError(env, absl::InvalidArgumentError(
                          "Packet does not contain mediapipe::Image."));
    return;
  }

  const mediapipe::Image& image = packet.Get<mediapipe::Image>();
  absl::Status status = segmenter->SetImage(image);
  ThrowIfError(env, status);
}

JNIEXPORT jlong JNICALL
Java_com_google_mediapipe_tasks_vision_interactivesegmenter_InteractiveSegmenter_nativeSegment(
    JNIEnv* env, jobject thiz, jlong segmenter_handle, jbyteArray strokes_bytes,
    jlong graph_handle) {
  InteractiveSegmenter* segmenter =
      reinterpret_cast<InteractiveSegmenter*>(segmenter_handle);

  jsize strokes_len = env->GetArrayLength(strokes_bytes);
  jbyte* strokes_data = env->GetByteArrayElements(strokes_bytes, nullptr);
  mediapipe::tasks::vision::interactive_segmenter::proto::Strokes strokes;
  bool strokes_parsed = strokes.ParseFromArray(strokes_data, strokes_len);
  env->ReleaseByteArrayElements(strokes_bytes, strokes_data, JNI_ABORT);

  if (!strokes_parsed) {
    ThrowIfError(env,
                 absl::InvalidArgumentError("Failed to parse Strokes proto"));
    return 0;
  }

  absl::StatusOr<mediapipe::Image> mask_or_status = segmenter->Segment(strokes);
  if (!mask_or_status.ok()) {
    ThrowIfError(env, mask_or_status.status());
    return 0;
  }

  auto result_packet = mediapipe::MakePacket<mediapipe::Image>(
      std::move(mask_or_status).value());
  auto* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(graph_handle);
  return mediapipe_graph->WrapPacketIntoContext(result_packet);
}

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_vision_interactivesegmenter_InteractiveSegmenter_nativeClose(
    JNIEnv* env, jobject thiz, jlong segmenter_handle) {
  InteractiveSegmenter* segmenter =
      reinterpret_cast<InteractiveSegmenter*>(segmenter_handle);
  delete segmenter;
}

}  // extern "C"
