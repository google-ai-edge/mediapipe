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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_SERVICE_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_SERVICE_JNI_H_

#include <jni.h>

#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/packet.h"

namespace mediapipe {
namespace android {

// Support class for handling graph services in JNI.
// It keeps the context argument opaque and avoids exposing the entire
// Graph to service JNI implementations.
class GraphServiceHelper {
 public:
  // Call this static method to provide a native service object in response to
  // a call to GraphService#installServiceObject in Java.
  // The context_handle parameter should be the same as passed to
  // installServiceObject.
  template <typename T>
  static void SetServiceObject(jlong context_handle,
                               const GraphService<T>& service,
                               std::shared_ptr<T> object) {
    SetServicePacket(context_handle, service,
                     MakePacket<std::shared_ptr<T>>(std::move(object)));
  }

 private:
  static void SetServicePacket(jlong context_handle,
                               const GraphServiceBase& service, Packet packet);
};

}  // namespace android
}  // namespace mediapipe

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_SERVICE_JNI_H_
