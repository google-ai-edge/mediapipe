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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_ANDROID_PACKET_CREATOR_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_ANDROID_PACKET_CREATOR_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define ANDROID_PACKET_CREATOR_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_AndroidPacketCreator_##METHOD_NAME

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(
    nativeCreateRgbImageFrame)(JNIEnv* env, jobject thiz, jlong context,
                               jobject bitmap);

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(
    nativeCreateRgbaImageFrame)(JNIEnv* env, jobject thiz, jlong context,
                                jobject bitmap);

JNIEXPORT jlong JNICALL ANDROID_PACKET_CREATOR_METHOD(nativeCreateRgbaImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject bitmap);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_ANDROID_PACKET_CREATOR_JNI_H_
