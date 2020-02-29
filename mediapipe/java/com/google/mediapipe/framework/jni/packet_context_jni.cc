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

#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_context_jni.h"

#include "absl/strings/str_format.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"

// Releases a native mediapipe packet.
JNIEXPORT void JNICALL PACKET_METHOD(nativeReleasePacket)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong packet) {
  // Removes the packet from the mediapipe context.
  mediapipe::android::Graph::RemovePacket(packet);
}

JNIEXPORT jlong JNICALL PACKET_METHOD(nativeGetTimestamp)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong packet) {
  return mediapipe::android::Graph::GetPacketFromHandle(packet)
      .Timestamp()
      .Value();
}

JNIEXPORT jlong JNICALL PACKET_METHOD(nativeCopyPacket)(JNIEnv* env,
                                                        jobject thiz,
                                                        jlong packet) {
  auto mediapipe_graph =
      mediapipe::android::Graph::GetContextFromHandle(packet);
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  return mediapipe_graph->WrapPacketIntoContext(mediapipe_packet);
}

jobject CreateJavaPacket(JNIEnv* env, jclass packet_cls, jlong packet) {
  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();

  std::string packet_class_name = class_registry.GetClassName(
      mediapipe::android::ClassRegistry::kPacketClassName);
  std::string create_method_name = class_registry.GetMethodName(
      mediapipe::android::ClassRegistry::kPacketClassName, "create");

  std::string signature = absl::StrFormat("(J)L%s;", packet_class_name);
  jmethodID createMethod = env->GetStaticMethodID(
      packet_cls, create_method_name.c_str(), signature.c_str());
  return env->CallStaticObjectMethod(packet_cls, createMethod, packet);
}
