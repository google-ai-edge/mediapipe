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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_GETTER_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_GETTER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define PACKET_GETTER_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_PacketGetter_##METHOD_NAME

// Get a native mediapipe packet.
JNIEXPORT jlong JNICALL PACKET_GETTER_METHOD(nativeGetPacketFromReference)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jlongArray JNICALL PACKET_GETTER_METHOD(nativeGetPairPackets)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jlongArray JNICALL PACKET_GETTER_METHOD(nativeGetVectorPackets)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jshort JNICALL PACKET_GETTER_METHOD(nativeGetInt16)(JNIEnv* env,
                                                              jobject thiz,
                                                              jlong packet);

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetInt32)(JNIEnv* env,
                                                            jobject thiz,
                                                            jlong packet);

JNIEXPORT jlong JNICALL PACKET_GETTER_METHOD(nativeGetInt64)(JNIEnv* env,
                                                             jobject thiz,
                                                             jlong packet);

JNIEXPORT jfloat JNICALL PACKET_GETTER_METHOD(nativeGetFloat32)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong packet);

JNIEXPORT jdouble JNICALL PACKET_GETTER_METHOD(nativeGetFloat64)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet);

JNIEXPORT jboolean JNICALL PACKET_GETTER_METHOD(nativeGetBool)(JNIEnv* env,
                                                               jobject thiz,
                                                               jlong packet);

JNIEXPORT jstring JNICALL PACKET_GETTER_METHOD(nativeGetString)(JNIEnv* env,
                                                                jobject thiz,
                                                                jlong packet);

JNIEXPORT jbyteArray JNICALL PACKET_GETTER_METHOD(nativeGetBytes)(JNIEnv* env,
                                                                  jobject thiz,
                                                                  jlong packet);

JNIEXPORT jbyteArray JNICALL PACKET_GETTER_METHOD(nativeGetProtoBytes)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT void JNICALL PACKET_GETTER_METHOD(nativeGetProto)(JNIEnv* env,
                                                            jobject thiz,
                                                            jlong packet,
                                                            jobject result);

JNIEXPORT jobjectArray JNICALL PACKET_GETTER_METHOD(nativeGetProtoVector)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jshortArray JNICALL PACKET_GETTER_METHOD(nativeGetInt16Vector)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jintArray JNICALL PACKET_GETTER_METHOD(nativeGetInt32Vector)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jlongArray JNICALL PACKET_GETTER_METHOD(nativeGetInt64Vector)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jfloatArray JNICALL PACKET_GETTER_METHOD(nativeGetFloat32Vector)(
    JNIEnv* env, jobject thiz, jlong packet);

JNIEXPORT jdoubleArray JNICALL PACKET_GETTER_METHOD(nativeGetFloat64Vector)(
    JNIEnv* env, jobject thiz, jlong packet);

// ImageFrame jni functions.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetImageWidth)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet);

JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetImageHeight)(JNIEnv* env,
                                                                  jobject thiz,
                                                                  jlong packet);

// Before calling this, the byte_buffer needs to have the correct allocated
// size.
JNIEXPORT jboolean JNICALL PACKET_GETTER_METHOD(nativeGetImageData)(
    JNIEnv* env, jobject thiz, jlong packet, jobject byte_buffer);

// Before calling this, the byte_buffer needs to have the correct allocated
// size.
JNIEXPORT jboolean JNICALL PACKET_GETTER_METHOD(nativeGetRgbaFromRgb)(
    JNIEnv* env, jobject thiz, jlong packet, jobject byte_buffer);

// Returns the width in VideoHeader packet.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetVideoHeaderWidth)(
    JNIEnv* env, jobject thiz, jlong packet);

// Returns the height in VideoHeader packet.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetVideoHeaderHeight)(
    JNIEnv* env, jobject thiz, jlong packet);

// Gets the byte array from the MediaPipe audio data..
JNIEXPORT jbyteArray JNICALL PACKET_GETTER_METHOD(nativeGetAudioData)(
    JNIEnv* env, jobject thiz, jlong packet);

// Gets number of channels in time series header packet.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(
    nativeGetTimeSeriesHeaderNumChannels)(JNIEnv* env, jobject thiz,
                                          jlong packet);

// Gets sampling rate in time series header packet.
JNIEXPORT jdouble JNICALL PACKET_GETTER_METHOD(
    nativeGetTimeSeriesHeaderSampleRate)(JNIEnv* env, jobject thiz,
                                         jlong packet);

// Returns the raw float array data for the MediaPipe Matrix.
// Note: MediaPipe::Matrix is column major matrix.
JNIEXPORT jfloatArray JNICALL PACKET_GETTER_METHOD(nativeGetMatrixData)(
    JNIEnv* env, jobject thiz, jlong packet);

// Returns the number of rows of the matrix.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetMatrixRows)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet);

// Returns the number of cols of the matrix.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetMatrixCols)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet);

// Returns the GL texture name of the mediapipe::GpuBuffer.
JNIEXPORT jint JNICALL PACKET_GETTER_METHOD(nativeGetGpuBufferName)(
    JNIEnv* env, jobject thiz, jlong packet);

// Returns a mediapipe::GlTextureBufferSharedPtr*.
// This will survive independently of the packet.
JNIEXPORT jlong JNICALL PACKET_GETTER_METHOD(nativeGetGpuBuffer)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong packet);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_PACKET_GETTER_JNI_H_
