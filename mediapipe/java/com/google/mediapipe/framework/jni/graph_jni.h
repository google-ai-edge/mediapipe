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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define GRAPH_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_Graph_##METHOD_NAME

// Creates a native mediapipe context.
JNIEXPORT jlong JNICALL GRAPH_METHOD(nativeCreateGraph)(JNIEnv* env,
                                                        jobject thiz);

// Releases a native mediapipe context.
JNIEXPORT void JNICALL GRAPH_METHOD(nativeReleaseGraph)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong context);

// Loads a binary mediapipe graph into the context.
JNIEXPORT void JNICALL GRAPH_METHOD(nativeLoadBinaryGraph)(JNIEnv* env,
                                                           jobject thiz,
                                                           jlong context,
                                                           jstring path);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeLoadBinaryGraphBytes)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeLoadBinaryGraphTemplate)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetGraphType)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong context,
                                                        jstring graph_type);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetGraphOptions)(JNIEnv* env,
                                                           jobject thiz,
                                                           jlong context,
                                                           jbyteArray data);

JNIEXPORT jbyteArray JNICALL GRAPH_METHOD(nativeGetCalculatorGraphConfig)(
    JNIEnv* env, jobject thiz, jlong context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeAddPacketCallback)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name,
    jobject callback);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeAddMultiStreamCallback)(
    JNIEnv* env, jobject thiz, jlong context, jobject stream_names,
    jobject callback, jboolean observe_timestamp_bounds);

JNIEXPORT jlong JNICALL GRAPH_METHOD(nativeAddSurfaceOutput)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeRunGraphUntilClose)(
    JNIEnv* env, jobject thiz, jlong context, jobjectArray stream_names,
    jlongArray packets);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeStartRunningGraph)(
    JNIEnv* env, jobject thiz, jlong context, jobjectArray side_packet_names,
    jlongArray side_packet_handles, jobjectArray stream_names_with_header,
    jlongArray header_handles);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeAddPacketToInputStream)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name, jlong packet,
    jlong timestamp);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeMovePacketToInputStream)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name, jlong packet,
    jlong timestamp);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetGraphInputStreamBlockingMode)(
    JNIEnv* env, jobject thiz, jlong context, jboolean mode);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCloseInputStream)(
    JNIEnv* env, jobject thiz, jlong context, jstring stream_name);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCloseAllInputStreams)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCloseAllPacketSources)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeWaitUntilGraphDone)(JNIEnv* env,
                                                              jobject thiz,
                                                              jlong context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeWaitUntilGraphIdle)(JNIEnv* env,
                                                              jobject thiz,
                                                              jlong context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeUpdatePacketReference)(
    JNIEnv* env, jobject thiz, jlong reference_packet, jlong new_packet);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeSetParentGlContext)(
    JNIEnv* env, jobject thiz, jlong context, jlong javaGlContext);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeCancelGraph)(JNIEnv* env,
                                                       jobject thiz,
                                                       jlong context);

JNIEXPORT jlong JNICALL GRAPH_METHOD(nativeGetProfiler)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_JNI_H_
