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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_CREATOR_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_CREATOR_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define PACKET_CREATOR_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_PacketCreator_##METHOD_NAME

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateReferencePacket)(
    JNIEnv* env, jobject thiz, jlong context, jlong packet);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloatImageFrame)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbaImageFrame)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbImageFromRgba)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGrayscaleImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateAudioPacketDirect)(
    JNIEnv* env, jobject thiz, jlong context, jobject data, jint num_channels,
    jint num_samples);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateAudioPacket)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data, jint offset,
    jint num_channels, jint num_samples);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt16)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jshort value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jint value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt64)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jlong value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32)(
    JNIEnv* env, jobject thiz, jlong context, jfloat value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat64)(
    JNIEnv* env, jobject thiz, jlong context, jdouble value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateBool)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong context,
                                                                jboolean value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateString)(
    JNIEnv* env, jobject thiz, jlong context, jstring value);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateVideoHeader)(
    JNIEnv* env, jobject thiz, jlong context, jint width, jint height);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateTimeSeriesHeader)(
    JNIEnv* env, jobject thiz, jlong context, jint num_channels,
    jdouble sample_rate);

// Creates a MediaPipe::Matrix packet using the float array data.
// The data must in column major order.
JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateMatrix)(
    JNIEnv* env, jobject thiz, jlong context, jint rows, jint cols,
    jfloatArray data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCpuImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height, jint num_channels);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGpuImage)(
    JNIEnv* env, jobject thiz, jlong context, jint name, jint width,
    jint height, jobject texture_release_callback);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGpuBuffer)(
    JNIEnv* env, jobject thiz, jlong context, jint name, jint width,
    jint height, jobject texture_release_callback);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32Array)(
    JNIEnv* env, jobject thiz, jlong context, jfloatArray data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32Vector)(
    JNIEnv* env, jobject thiz, jlong context, jfloatArray data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32Array)(
    JNIEnv* env, jobject thiz, jlong context, jintArray data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateStringFromByteArray)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCalculatorOptions)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateProto)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jobject data);

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCameraIntrinsics)(
    JNIEnv* env, jobject thiz, jlong context, jfloat fx, jfloat fy, jfloat cx,
    jfloat cy, jfloat width, jfloat height);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_CREATOR_JNI_H_
