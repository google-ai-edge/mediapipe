/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_SOURCE_OR_NODE_OUTPUT_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_SOURCE_OR_NODE_OUTPUT_H_

#include "mediapipe/framework/api2/builder.h"

namespace mediapipe {
namespace tasks {

// Helper class representing either a Source object or a GenericNode output.
//
// Source and MultiSource (the output of a GenericNode) are widely incompatible,
// but being able to represent either of these in temporary variables and
// connect them later on facilitates graph building.
template <typename T>
class SourceOrNodeOutput {
 public:
  SourceOrNodeOutput() = delete;
  // The caller is responsible for ensuring 'source' outlives this object.
  explicit SourceOrNodeOutput(mediapipe::api2::builder::Source<T>* source)
      : source_(source) {}
  // The caller is responsible for ensuring 'node' outlives this object.
  SourceOrNodeOutput(mediapipe::api2::builder::GenericNode* node,
                     std::string tag)
      : node_(node), tag_(tag) {}
  // The caller is responsible for ensuring 'node' outlives this object.
  SourceOrNodeOutput(mediapipe::api2::builder::GenericNode* node, int index)
      : node_(node), index_(index) {}

  // Connects the source or node output to the provided destination.
  template <typename U>
  void operator>>(const U& dest) {
    if (source_ != nullptr) {
      *source_ >> dest;
    } else {
      if (index_ < 0) {
        node_->Out(tag_) >> dest;
      } else {
        node_->Out(index_) >> dest;
      }
    }
  }

 private:
  mediapipe::api2::builder::Source<T>* source_ = nullptr;
  mediapipe::api2::builder::GenericNode* node_ = nullptr;
  std::string tag_ = "";
  int index_ = -1;
};

}  // namespace tasks
}  // namespace mediapipe
#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_SOURCE_OR_NODE_OUTPUT_H_
