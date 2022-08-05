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

#include "absl/container/node_hash_map.h"
#include <jni.h>
#include <string>

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
    std::string GetFieldName(std::string cls, std::string field);

    // TODO: Just have the prefix instead of all these constants.
    static constexpr const char* kAndroidAssetUtilClassName =
        "com/google/mediapipe/framework/AndroidAssetUtil";
    static constexpr const char* kAndroidPacketCreatorClassName =
        "com/google/mediapipe/framework/AndroidPacketCreator";
    static constexpr const char* kCompatClassName =
        "com/google/mediapipe/framework/Compat";
    static constexpr const char* kGraphClassName =
        "com/google/mediapipe/framework/Graph";
    static constexpr const char* kGraphProfilerClassName =
        "com/google/mediapipe/framework/GraphProfiler";
    static constexpr const char* kPacketClassName =
        "com/google/mediapipe/framework/Packet";
    static constexpr const char* kMediaPipeExceptionClassName =
        "com/google/mediapipe/framework/MediaPipeException";
    static constexpr const char* kPacketCallbackClassName =
        "com/google/mediapipe/framework/PacketCallback";
    static constexpr const char* kPacketListCallbackClassName =
        "com/google/mediapipe/framework/PacketListCallback";
    static constexpr const char* kPacketCreatorClassName =
        "com/google/mediapipe/framework/PacketCreator";
    static constexpr const char* kPacketGetterClassName =
        "com/google/mediapipe/framework/PacketGetter";
    static constexpr const char* kPacketWithHeaderCallbackClassName =
        "com/google/mediapipe/framework/PacketWithHeaderCallback";
    static constexpr const char* kProtoUtilSerializedMessageClassName =
        "com/google/mediapipe/framework/ProtoUtil$SerializedMessage";

private:
    ClassRegistry();
    absl::node_hash_map<std::string, std::string> renaming_map_;
};

}  // namespace android
}  // namespace mediapipe

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_CLASS_REGISTRY_H_
