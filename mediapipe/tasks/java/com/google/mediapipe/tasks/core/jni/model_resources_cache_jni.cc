// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/model_resources_cache_jni.h"

#include <utility>

#include "mediapipe/java/com/google/mediapipe/framework/jni/graph_service_jni.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "tensorflow/lite/kernels/register.h"

namespace {
using ::mediapipe::tasks::core::kModelResourcesCacheService;
using ::mediapipe::tasks::core::MediaPipeBuiltinOpResolver;
using ::mediapipe::tasks::core::ModelResourcesCache;
using HandleType = std::shared_ptr<ModelResourcesCache>*;
}  // namespace

JNIEXPORT jlong JNICALL MODEL_RESOURCES_CACHE_METHOD(
    nativeCreateModelResourcesCache)(JNIEnv* env, jobject thiz) {
  auto ptr = std::make_shared<ModelResourcesCache>(
      absl::make_unique<MediaPipeBuiltinOpResolver>());
  HandleType handle = new std::shared_ptr<ModelResourcesCache>(std::move(ptr));
  return reinterpret_cast<jlong>(handle);
}

JNIEXPORT void JNICALL MODEL_RESOURCES_CACHE_METHOD(
    nativeReleaseModelResourcesCache)(JNIEnv* env, jobject thiz,
                                      jlong nativeHandle) {
  delete reinterpret_cast<HandleType>(nativeHandle);
}

JNIEXPORT void JNICALL MODEL_RESOURCES_CACHE_SERVICE_METHOD(
    nativeInstallServiceObject)(JNIEnv* env, jobject thiz, jlong contextHandle,
                                jlong objectHandle) {
  mediapipe::android::GraphServiceHelper::SetServiceObject(
      contextHandle, kModelResourcesCacheService,
      *reinterpret_cast<HandleType>(objectHandle));
}
