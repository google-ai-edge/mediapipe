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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_CONTEXT_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_CONTEXT_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define PACKET_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_Packet_##METHOD_NAME

// Releases a native mediapipe packet.
JNIEXPORT void JNICALL PACKET_METHOD(nativeReleasePacket)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong packet);

// Returns the timestamp of the packet.
JNIEXPORT jlong JNICALL PACKET_METHOD(nativeGetTimestamp)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong packet);

// Returns true if the packet is empty.
JNIEXPORT jboolean JNICALL PACKET_METHOD(nativeIsEmpty)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong packet);

// Make a copy of a mediapipe packet, basically increase the reference count.
JNIEXPORT jlong JNICALL PACKET_METHOD(nativeCopyPacket)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong packet);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

// Calls the java method to create an instance of java Packet.
jobject CreateJavaPacket(JNIEnv* env, jclass packet_cls, jlong packet);

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_CONTEXT_JNI_H_
