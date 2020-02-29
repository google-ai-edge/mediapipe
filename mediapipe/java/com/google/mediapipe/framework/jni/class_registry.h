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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_CLASS_REGISTRY_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_CLASS_REGISTRY_H_

#include <jni.h>

#include <string>

#include "absl/container/node_hash_map.h"

namespace mediapipe {
namespace android {

// ClassRegistry maintains the correct names of Java classes and methods and
// should be queried before any calls to FindClass() or GetMethodID().
class ClassRegistry {
 public:
  static ClassRegistry& GetInstance();
  void InstallRenamingMap(
      absl::node_hash_map<std::string, std::string> renaming_map);
  std::string GetClassName(std::string cls);
  std::string GetMethodName(std::string cls, std::string method);

  // TODO: Just have the prefix instead of all these constants.
  static constexpr char const* kAndroidAssetUtilClassName =
      "com/google/mediapipe/framework/AndroidAssetUtil";
  static constexpr char const* kAndroidPacketCreatorClassName =
      "com/google/mediapipe/framework/AndroidPacketCreator";
  static constexpr char const* kCompatClassName =
      "com/google/mediapipe/framework/Compat";
  static constexpr char const* kGraphClassName =
      "com/google/mediapipe/framework/Graph";
  static constexpr char const* kPacketClassName =
      "com/google/mediapipe/framework/Packet";
  static constexpr char const* kMediaPipeExceptionClassName =
      "com/google/mediapipe/framework/MediaPipeException";
  static constexpr char const* kPacketCallbackClassName =
      "com/google/mediapipe/framework/PacketCallback";
  static constexpr char const* kPacketCreatorClassName =
      "com/google/mediapipe/framework/PacketCreator";
  static constexpr char const* kPacketGetterClassName =
      "com/google/mediapipe/framework/PacketGetter";
  static constexpr char const* kPacketWithHeaderCallbackClassName =
      "com/google/mediapipe/framework/PacketWithHeaderCallback";

 private:
  ClassRegistry();
  absl::node_hash_map<std::string, std::string> renaming_map_;
};

}  // namespace android
}  // namespace mediapipe

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_CLASS_REGISTRY_H_
