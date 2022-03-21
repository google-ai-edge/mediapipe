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

#include "mediapipe/java/com/google/mediapipe/framework/jni/graph_jni.h"

#include <memory>
#include <string>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"

using mediapipe::android::JavaListToStdStringVector;
using mediapipe::android::JStringToStdString;
using mediapipe::android::ThrowIfError;

namespace {
absl::Status AddSidePacketsIntoGraph(mediapipe::android::Graph* mediapipe_graph,
                                     JNIEnv* env, jobjectArray stream_names,
                                     jlongArray packets) {
  jsize num_side_packets = env->GetArrayLength(stream_names);
  if (num_side_packets != env->GetArrayLength(packets)) {
    return absl::InvalidArgumentError(
        "Number of streams and packets doesn't match!");
  }
  // Note, packets_array_ref is really a const jlong* but this clashes with the
  // the expectation of ReleaseLongArrayElements below.
  jlong* packets_array_ref = env->GetLongArrayElements(packets, nullptr);
  for (jsize i = 0; i < num_side_packets; ++i) {
    jstring name =
        reinterpret_cast<jstring>(env->GetObjectArrayElement(stream_names, i));
    mediapipe_graph->SetInputSidePacket(
        JStringToStdString(env, name),
        mediapipe::android::Graph::GetPacketFromHandle(packets_array_ref[i]));
    env->DeleteLocalRef(name);
  }
  env->ReleaseLongArrayElements(packets, packets_array_ref, JNI_ABORT);
  return absl::OkStatus();
}

absl::Status AddStreamHeadersIntoGraph(
    mediapipe::android::Graph* mediapipe_graph, JNIEnv* env,
    jobjectArray stream_names, jlongArray packets) {
  jsize num_headers = env->GetArrayLength(stream_names);
  if (num_headers != env->GetArrayLength(packets)) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Number of streams and packets doesn't match!");
  }
  jlong* packets_array_ref = env->GetLongArrayElements(packets, nullptr);
  for (jsize i = 0; i < num_headers; ++i) {
    jstring name =
        reinterpret_cast<jstring>(env->GetObjectArrayElement(stream_names, i));
    mediapipe_graph->SetStreamHeader(
        JStringToStdString(env, name),
        mediapipe::android::Graph::GetPacketFromHandle(packets_array_ref[i]));
    env->DeleteLocalRef(name);
  }
  env->ReleaseLongArrayElements(packets, packets_array_ref, JNI_ABORT);
  return absl::OkStatus();
}

}  // namespace

JNIEXPORT jlong JNICALL GRAPH_METHOD(nativeCreateGraph)(JNIEnv* env,
                                                        jobject thiz) {
  if (!mediapipe::java::SetJavaVM(env)) {
    return 0;
  }
  return reinterpret_cast<int64_t>(new mediapipe::android::Graph());
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeReleaseGraph)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong context) {
  delete reinterpret_cast<mediapipe::android::Graph*>(context);
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeLoadBinaryGraph)(JNIEnv* env,
                                                           jobject thiz,
                                                           jlong context,
                                                           jstring path) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  const char* path_ref = env->GetStringUTFChars(path, nullptr);
  // Make a copy of the string and release the jni reference.
  std::string path_to_graph(path_ref);
  env->ReleaseStringUTFChars(path, path_ref);
  ThrowIfError(env, mediapipe_graph->LoadBinaryGraph(path_to_graph));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeLoadBinaryGraphBytes)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  jbyte* data_ptr = env->GetByteArrayElements(data, nullptr);
  int size = env->GetArrayLength(data);
  absl::Status status =
      mediapipe_graph->LoadBinaryGraph(reinterpret_cast<char*>(data_ptr), size);
  env->ReleaseByteArrayElements(data, data_ptr, JNI_ABORT);
  ThrowIfError(env, status);
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeLoadBinaryGraphTemplate)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  jbyte* data_ptr = env->GetByteArrayElements(data, nullptr);
  int size = env->GetArrayLength(data);
  absl::Status status = mediapipe_graph->LoadBinaryGraphTemplate(
      reinterpret_cast<char*>(data_ptr), size);
  env->ReleaseByteArrayElements(data, data_ptr, JNI_ABORT);
  ThrowIfError(env, status);
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetGraphType)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong context,
                                                        jstring graph_type) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  const char* graph_type_ref = env->GetStringUTFChars(graph_type, nullptr);
  // Make a copy of the string and release the jni reference.
  std::string graph_type_string(graph_type_ref);
  env->ReleaseStringUTFChars(graph_type, graph_type_ref);
  ThrowIfError(env, mediapipe_graph->SetGraphType(graph_type_string));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetGraphOptions)(JNIEnv* env,
                                                           jobject thiz,
                                                           jlong context,
                                                           jbyteArray data) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  jbyte* data_ptr = env->GetByteArrayElements(data, nullptr);
  int size = env->GetArrayLength(data);
  absl::Status status =
      mediapipe_graph->SetGraphOptions(reinterpret_cast<char*>(data_ptr), size);
  env->ReleaseByteArrayElements(data, data_ptr, JNI_ABORT);
  ThrowIfError(env, status);
}

JNIEXPORT jbyteArray JNICALL GRAPH_METHOD(nativeGetCalculatorGraphConfig)(
    JNIEnv* env, jobject thiz, jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  auto graph = mediapipe_graph->GetCalculatorGraphConfig();
  if (graph.IsInitialized()) {
    int size = graph.ByteSize();
    char* buffer = new char[size];
    graph.SerializeToArray(buffer, size);
    jbyteArray byteArray = env->NewByteArray(size);
    env->SetByteArrayRegion(byteArray, 0, size,
                            reinterpret_cast<jbyte*>(buffer));
    return byteArray;
  }
  return nullptr;
}

JNIEXPORT void JNICALL
GRAPH_METHOD(nativeAddPacketCallback)(JNIEnv* env, jobject thiz, jlong context,
                                      jstring stream_name, jobject callback) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  std::string output_stream_name = JStringToStdString(env, stream_name);

  // Create a global reference to the callback object, so that it can
  // be accessed later.
  jobject global_callback_ref = env->NewGlobalRef(callback);
  if (!global_callback_ref) {
    ThrowIfError(env,
                 absl::InternalError("Failed to allocate packet callback"));
    return;
  }
  ThrowIfError(env, mediapipe_graph->AddCallbackHandler(output_stream_name,
                                                        global_callback_ref));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeAddMultiStreamCallback)(
    JNIEnv* env, jobject thiz, jlong context, jobject stream_names,
    jobject callback, jboolean observe_timestamp_bounds) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  std::vector<std::string> output_stream_names =
      JavaListToStdStringVector(env, stream_names);
  for (const std::string& s : output_stream_names) {
    if (s.empty()) {
      ThrowIfError(env,
                   absl::InternalError("streamNames is not correctly parsed or "
                                       "it contains empty string."));
      return;
    }
  }

  // Create a global reference to the callback object, so that it can
  // be accessed later.
  jobject global_callback_ref = env->NewGlobalRef(callback);
  if (!global_callback_ref) {
    ThrowIfError(env,
                 absl::InternalError("Failed to allocate packets callback"));
    return;
  }
  ThrowIfError(env, mediapipe_graph->AddMultiStreamCallbackHandler(
                        output_stream_names, global_callback_ref,
                        observe_timestamp_bounds));
}

JNIEXPORT jlong JNICALL GRAPH_METHOD(nativeAddSurfaceOutput)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  std::string output_stream_name = JStringToStdString(env, stream_name);

  return mediapipe_graph->AddSurfaceOutput(output_stream_name);
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeRunGraphUntilClose)(
    JNIEnv* env, jobject thiz, jlong context, jobjectArray stream_names,
    jlongArray packets) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  if (ThrowIfError(env, AddSidePacketsIntoGraph(mediapipe_graph, env,
                                                stream_names, packets))) {
    return;
  }
  ThrowIfError(env, mediapipe_graph->RunGraphUntilClose(env));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeStartRunningGraph)(
    JNIEnv* env, jobject thiz, jlong context, jobjectArray side_packet_names,
    jlongArray side_packet_handles, jobjectArray stream_names_with_header,
    jlongArray header_handles) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  if (ThrowIfError(
          env, AddSidePacketsIntoGraph(mediapipe_graph, env, side_packet_names,
                                       side_packet_handles))) {
    return;
  }
  if (ThrowIfError(env, AddStreamHeadersIntoGraph(mediapipe_graph, env,
                                                  stream_names_with_header,
                                                  header_handles))) {
    return;
  }
  ThrowIfError(env, mediapipe_graph->StartRunningGraph(env));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeAddPacketToInputStream)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name, jlong packet,
    jlong timestamp) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  // We push in a copy of the current packet at the given timestamp.
  ThrowIfError(
      env, mediapipe_graph->AddPacketToInputStream(
               JStringToStdString(env, stream_name),
               mediapipe::android::Graph::GetPacketFromHandle(packet).At(
                   mediapipe::Timestamp::CreateNoErrorChecking(timestamp))));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeMovePacketToInputStream)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name, jlong packet,
    jlong timestamp) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);

  ThrowIfError(
      env, mediapipe_graph->SetTimestampAndMovePacketToInputStream(
               JStringToStdString(env, stream_name),
               static_cast<int64_t>(packet), static_cast<int64_t>(timestamp)));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetGraphInputStreamBlockingMode)(
    JNIEnv* env, jobject thiz, jlong context, jboolean mode) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  if (mode) {
    mediapipe_graph->SetGraphInputStreamAddMode(
        mediapipe::CalculatorGraph::GraphInputStreamAddMode::
            WAIT_TILL_NOT_FULL);
  } else {
    mediapipe_graph->SetGraphInputStreamAddMode(
        mediapipe::CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
  }
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCloseInputStream)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  ThrowIfError(env, mediapipe_graph->CloseInputStream(
                        JStringToStdString(env, stream_name)));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCloseAllInputStreams)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  ThrowIfError(env, mediapipe_graph->CloseAllInputStreams());
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCloseAllPacketSources)(
    JNIEnv* env, jobject thiz, jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  ThrowIfError(env, mediapipe_graph->CloseAllPacketSources());
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeWaitUntilGraphDone)(JNIEnv* env,
                                                              jobject thiz,
                                                              jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  ThrowIfError(env, mediapipe_graph->WaitUntilDone(env));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeWaitUntilGraphIdle)(JNIEnv* env,
                                                              jobject thiz,
                                                              jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  ThrowIfError(env, mediapipe_graph->WaitUntilIdle(env));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeUpdatePacketReference)(
    JNIEnv* env, jobject thiz, jlong reference_packet, jlong new_packet) {
  auto reference =
      mediapipe::android::Graph::GetPacketFromHandle(reference_packet)
          .Get<std::unique_ptr<mediapipe::SyncedPacket>>()
          .get();
  auto new_value = mediapipe::android::Graph::GetPacketFromHandle(new_packet);
  reference->UpdatePacket(new_value);
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetParentGlContext)(
    JNIEnv* env, jobject thiz, jlong context, jlong javaGlContext) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  ThrowIfError(env, mediapipe_graph->SetParentGlContext(javaGlContext));
}

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCancelGraph)(JNIEnv* env,
                                                       jobject thiz,
                                                       jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  mediapipe_graph->CancelGraph();
}

JNIEXPORT jlong JNICALL GRAPH_METHOD(nativeGetProfiler)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong context) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  return reinterpret_cast<jlong>(mediapipe_graph->GetProfilingContext());
}
