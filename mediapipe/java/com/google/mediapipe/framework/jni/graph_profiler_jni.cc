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

#include "mediapipe/java/com/google/mediapipe/framework/jni/graph_profiler_jni.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_profile.pb.h"

JNIEXPORT void JNICALL GRAPH_PROFILER_METHOD(nativeReset)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong handle) {
  mediapipe::ProfilingContext* profiling_context =
      reinterpret_cast<mediapipe::ProfilingContext*>(handle);
  profiling_context->Reset();
}

JNIEXPORT void JNICALL GRAPH_PROFILER_METHOD(nativePause)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong handle) {
  mediapipe::ProfilingContext* profiling_context =
      reinterpret_cast<mediapipe::ProfilingContext*>(handle);
  profiling_context->Pause();
}

JNIEXPORT void JNICALL GRAPH_PROFILER_METHOD(nativeResume)(JNIEnv* env,
                                                           jobject thiz,
                                                           jlong handle) {
  mediapipe::ProfilingContext* profiling_context =
      reinterpret_cast<mediapipe::ProfilingContext*>(handle);
  profiling_context->Resume();
}

JNIEXPORT jobjectArray JNICALL GRAPH_PROFILER_METHOD(
    nativeGetCalculatorProfiles)(JNIEnv* env, jobject thiz, jlong handle) {
  mediapipe::ProfilingContext* profiling_context =
      reinterpret_cast<mediapipe::ProfilingContext*>(handle);

  std::vector<mediapipe::CalculatorProfile> profiles_vec;
  if (profiling_context->GetCalculatorProfiles(&profiles_vec) !=
      absl::OkStatus()) {
    return nullptr;
  }
  int num_profiles = profiles_vec.size();
  if (num_profiles == 0) {
    return nullptr;
  }

  // TODO: move to register natives.
  jclass byte_array_cls = env->FindClass("[B");
  jobjectArray profiles =
      env->NewObjectArray(num_profiles, byte_array_cls, nullptr);
  env->DeleteLocalRef(byte_array_cls);
  for (int i = 0; i < num_profiles; i++) {
    const auto& profile = profiles_vec[i];
    int size = profile.ByteSize();

    jbyteArray byteArray = env->NewByteArray(size);
    jbyte* byteArrayBuffer = env->GetByteArrayElements(byteArray, nullptr);
    profile.SerializeToArray(byteArrayBuffer, size);
    env->ReleaseByteArrayElements(byteArray, byteArrayBuffer, 0);

    env->SetObjectArrayElement(profiles, i, byteArray);
    env->DeleteLocalRef(byteArray);
  }

  return profiles;
}
