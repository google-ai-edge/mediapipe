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

#include "mediapipe/java/com/google/mediapipe/framework/jni/register_natives.h"

#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"

#if defined(__ANDROID__)
#include "mediapipe/java/com/google/mediapipe/framework/jni/android_asset_util_jni.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/android_packet_creator_jni.h"
#endif
#include "mediapipe/java/com/google/mediapipe/framework/jni/compat_jni.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph_jni.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_context_jni.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_creator_jni.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_getter_jni.h"

namespace mediapipe {
namespace android {
namespace registration {
namespace {

// TODO: Ideally all these methods would live in their own JNI files.
// We should have a JniOnLoadRegistry which collects a series of function ptrs
// to call when JNI_OnLoad is called. Each module would add its own hook with a
// static initializer.

struct JNINativeMethodStrings {
  std::string name;
  std::string signature;
  void *fnPtr;
};

void AddJNINativeMethod(std::vector<JNINativeMethodStrings> *methods,
                        std::string cls, std::string method,
                        std::string signature, void *fn) {
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string method_name = class_registry.GetMethodName(cls, method);
  if (method_name != method) {
    JNINativeMethodStrings jniNativeMethod{method_name, signature, fn};
    methods->push_back(jniNativeMethod);
  }
}

void RegisterNativesVector(JNIEnv *env, jclass cls,
                           const std::vector<JNINativeMethodStrings> &methods) {
  // A client Java project may not use some methods and classes that we attempt
  // to register and could be removed by Proguard. In that case, we want to
  // avoid triggering a crash due to ClassNotFoundException, so we are trading
  // safety check here in exchange for flexibility to list out all registrations
  // without worrying about usage subset by client Java projects.
  if (!cls || methods.empty()) {
    LOG(INFO) << "Skipping registration and clearing exception. Class or "
                 "native methods not found, may be unused and/or trimmed by "
                 "Proguard.";
    env->ExceptionClear();
    return;
  }

  JNINativeMethod *methods_array = new JNINativeMethod[methods.size()];
  for (int i = 0; i < methods.size(); i++) {
    JNINativeMethod jniNativeMethod{
        const_cast<char *>(methods[i].name.c_str()),
        const_cast<char *>(methods[i].signature.c_str()), methods[i].fnPtr};
    methods_array[i] = jniNativeMethod;
  }
  env->RegisterNatives(cls, methods_array, methods.size());
  delete[] methods_array;
}

void RegisterGraphNatives(JNIEnv *env) {
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string graph(mediapipe::android::ClassRegistry::kGraphClassName);
  std::string graph_name = class_registry.GetClassName(graph);
  jclass graph_class = env->FindClass(graph_name.c_str());

  std::vector<JNINativeMethodStrings> graph_methods;
  AddJNINativeMethod(&graph_methods, graph, "nativeCreateGraph", "()J",
                     (void *)&GRAPH_METHOD(nativeCreateGraph));
  AddJNINativeMethod(&graph_methods, graph, "nativeLoadBinaryGraph",
                     "(JLjava/lang/String;)V",
                     (void *)&GRAPH_METHOD(nativeLoadBinaryGraph));
  AddJNINativeMethod(&graph_methods, graph, "nativeLoadBinaryGraphBytes",
                     "(J[B)V",
                     (void *)&GRAPH_METHOD(nativeLoadBinaryGraphBytes));
  std::string packet_callback_name = class_registry.GetClassName(
      mediapipe::android::ClassRegistry::kPacketCallbackClassName);
  std::string native_add_packet_callback_signature =
      absl::StrFormat("(JLjava/lang/String;L%s;)V", packet_callback_name);
  AddJNINativeMethod(&graph_methods, graph, "nativeAddPacketCallback",
                     native_add_packet_callback_signature.c_str(),
                     (void *)&GRAPH_METHOD(nativeAddPacketCallback));
  AddJNINativeMethod(&graph_methods, graph, "nativeMovePacketToInputStream",
                     "(JLjava/lang/String;JJ)V",
                     (void *)&GRAPH_METHOD(nativeMovePacketToInputStream));
  AddJNINativeMethod(&graph_methods, graph, "nativeStartRunningGraph",
                     "(J[Ljava/lang/String;[J[Ljava/lang/String;[J)V",
                     (void *)&GRAPH_METHOD(nativeStartRunningGraph));
  AddJNINativeMethod(&graph_methods, graph, "nativeSetParentGlContext", "(JJ)V",
                     (void *)&GRAPH_METHOD(nativeSetParentGlContext));
  AddJNINativeMethod(&graph_methods, graph, "nativeCloseAllPacketSources",
                     "(J)V",
                     (void *)&GRAPH_METHOD(nativeCloseAllPacketSources));
  AddJNINativeMethod(&graph_methods, graph, "nativeWaitUntilGraphDone", "(J)V",
                     (void *)&GRAPH_METHOD(nativeWaitUntilGraphDone));
  AddJNINativeMethod(&graph_methods, graph, "nativeReleaseGraph", "(J)V",
                     (void *)&GRAPH_METHOD(nativeReleaseGraph));
  RegisterNativesVector(env, graph_class, graph_methods);
}

void RegisterAndroidAssetUtilNatives(JNIEnv *env) {
#if defined(__ANDROID__)
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string android_asset_util(
      mediapipe::android::ClassRegistry::kAndroidAssetUtilClassName);
  std::string android_asset_util_name =
      class_registry.GetClassName(android_asset_util);
  jclass android_asset_util_class =
      env->FindClass(android_asset_util_name.c_str());

  std::vector<JNINativeMethodStrings> android_asset_util_methods;
  AddJNINativeMethod(
      &android_asset_util_methods, android_asset_util,
      "nativeInitializeAssetManager",
      "(Landroid/content/Context;Ljava/lang/String;)Z",
      (void *)&ANDROID_ASSET_UTIL_METHOD(nativeInitializeAssetManager));
  RegisterNativesVector(env, android_asset_util_class,
                        android_asset_util_methods);
#endif
}

void RegisterAndroidPacketCreatorNatives(JNIEnv *env) {
#if defined(__ANDROID__)
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string android_packet_creator(
      mediapipe::android::ClassRegistry::kAndroidPacketCreatorClassName);
  std::string android_packet_creator_name =
      class_registry.GetClassName(android_packet_creator);
  jclass android_packet_creator_class =
      env->FindClass(android_packet_creator_name.c_str());

  std::vector<JNINativeMethodStrings> android_packet_creator_methods;
  AddJNINativeMethod(
      &android_packet_creator_methods, android_packet_creator,
      "nativeCreateRgbImageFrame", "(JLandroid/graphics/Bitmap;)J",
      (void *)&ANDROID_PACKET_CREATOR_METHOD(nativeCreateRgbImageFrame));
  RegisterNativesVector(env, android_packet_creator_class,
                        android_packet_creator_methods);
#endif
}

void RegisterPacketCreatorNatives(JNIEnv *env) {
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string packet_creator(
      mediapipe::android::ClassRegistry::kPacketCreatorClassName);
  std::string packet_creator_name = class_registry.GetClassName(packet_creator);
  jclass packet_creator_class = env->FindClass(packet_creator_name.c_str());

  std::vector<JNINativeMethodStrings> packet_creator_methods;
  AddJNINativeMethod(&packet_creator_methods, packet_creator,
                     "nativeCreateRgbImage", "(JLjava/nio/ByteBuffer;II)J",
                     (void *)&PACKET_CREATOR_METHOD(nativeCreateRgbImage));
  AddJNINativeMethod(
      &packet_creator_methods, packet_creator, "nativeCreateRgbaImageFrame",
      "(JLjava/nio/ByteBuffer;II)J",
      (void *)&PACKET_CREATOR_METHOD(nativeCreateRgbaImageFrame));
  AddJNINativeMethod(
      &packet_creator_methods, packet_creator, "nativeCreateFloatImageFrame",
      "(JLjava/nio/ByteBuffer;II)J",
      (void *)&PACKET_CREATOR_METHOD(nativeCreateFloatImageFrame));
  AddJNINativeMethod(&packet_creator_methods, packet_creator,
                     "nativeCreateInt32", "(JI)J",
                     (void *)&PACKET_CREATOR_METHOD(nativeCreateInt32));
  RegisterNativesVector(env, packet_creator_class, packet_creator_methods);
}

void RegisterPacketGetterNatives(JNIEnv *env) {
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string packet_getter(
      mediapipe::android::ClassRegistry::kPacketGetterClassName);
  std::string packet_getter_name = class_registry.GetClassName(packet_getter);
  jclass packet_getter_class = env->FindClass(packet_getter_name.c_str());

  std::vector<JNINativeMethodStrings> packet_getter_methods;
  AddJNINativeMethod(&packet_getter_methods, packet_getter, "nativeGetBytes",
                     "(J)[B", (void *)&PACKET_GETTER_METHOD(nativeGetBytes));
  AddJNINativeMethod(&packet_getter_methods, packet_getter,
                     "nativeGetProtoBytes", "(J)[B",
                     (void *)&PACKET_GETTER_METHOD(nativeGetProtoBytes));
  AddJNINativeMethod(&packet_getter_methods, packet_getter,
                     "nativeGetImageData", "(JLjava/nio/ByteBuffer;)Z",
                     (void *)&PACKET_GETTER_METHOD(nativeGetImageData));
  AddJNINativeMethod(&packet_getter_methods, packet_getter,
                     "nativeGetFloat32Vector", "(J)[F",
                     (void *)&PACKET_GETTER_METHOD(nativeGetFloat32Vector));
  RegisterNativesVector(env, packet_getter_class, packet_getter_methods);
}

void RegisterPacketNatives(JNIEnv *env) {
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string packet(mediapipe::android::ClassRegistry::kPacketClassName);
  std::string packet_name = class_registry.GetClassName(packet);
  jclass packet_class = env->FindClass(packet_name.c_str());

  std::vector<JNINativeMethodStrings> packet_methods;
  AddJNINativeMethod(&packet_methods, packet, "nativeReleasePacket", "(J)V",
                     (void *)&PACKET_METHOD(nativeReleasePacket));
  AddJNINativeMethod(&packet_methods, packet, "nativeCopyPacket", "(J)J",
                     (void *)&PACKET_METHOD(nativeCopyPacket));
  AddJNINativeMethod(&packet_methods, packet, "nativeGetTimestamp", "(J)J",
                     (void *)&PACKET_METHOD(nativeGetTimestamp));
  RegisterNativesVector(env, packet_class, packet_methods);
}

void RegisterCompatNatives(JNIEnv *env) {
  auto &class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string compat(mediapipe::android::ClassRegistry::kCompatClassName);
  std::string compat_name = class_registry.GetClassName(compat);
  jclass compat_class = env->FindClass(compat_name.c_str());

  std::vector<JNINativeMethodStrings> compat_methods;
  AddJNINativeMethod(&compat_methods, compat, "getCurrentNativeEGLContext",
                     "()J", (void *)&COMPAT_METHOD(getCurrentNativeEGLContext));
  AddJNINativeMethod(&compat_methods, compat, "getCurrentNativeEGLSurface",
                     "(I)J",
                     (void *)&COMPAT_METHOD(getCurrentNativeEGLSurface));
  RegisterNativesVector(env, compat_class, compat_methods);
}

}  // namespace

void RegisterAllNatives(JNIEnv *env) {
  RegisterGraphNatives(env);
  RegisterAndroidAssetUtilNatives(env);
  RegisterAndroidPacketCreatorNatives(env);
  RegisterPacketCreatorNatives(env);
  RegisterPacketGetterNatives(env);
  RegisterPacketNatives(env);
  RegisterCompatNatives(env);
}

}  // namespace registration
}  // namespace android
}  // namespace mediapipe
