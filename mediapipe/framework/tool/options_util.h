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

#ifndef MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_UTIL_H_

#include <typeindex>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/port/any_proto.h"

namespace mediapipe {

namespace tool {

// Combine a base options with an overriding options.
template <typename T>
inline T MergeOptions(const T& base, const T& options) {
  auto result = base;
  result.MergeFrom(options);
  return result;
}

// A compile-time detector for the constant |T::ext|.
template <typename T>
struct IsExtension {
 private:
  template <typename U>
  static char test(decltype(&U::ext));

  template <typename>
  static int test(...);

 public:
  static constexpr bool value = (sizeof(test<T>(0)) == sizeof(char));
};

// A map from object type to object.
class TypeMap {
 public:
  template <class T>
  bool Has() const {
    return content_.count(typeid(T)) > 0;
  }
  template <class T>
  T* Get() const {
    if (!Has<T>()) {
      content_[typeid(T)] = std::make_shared<T>();
    }
    return static_cast<T*>(content_[typeid(T)].get());
  }

 private:
  mutable std::map<std::type_index, std::shared_ptr<void>> content_;
};

template <class T,
          typename std::enable_if<IsExtension<T>::value, int>::type = 0>
void GetExtension(const CalculatorOptions& options, T* result) {
  if (options.HasExtension(T::ext)) {
    *result = options.GetExtension(T::ext);
  }
}

template <class T,
          typename std::enable_if<!IsExtension<T>::value, int>::type = 0>
void GetExtension(const CalculatorOptions& options, T* result) {}

template <class T>
void GetNodeOptions(const CalculatorGraphConfig::Node& node_config, T* result) {
#if defined(MEDIAPIPE_PROTO_LITE) && defined(MEDIAPIPE_PROTO_THIRD_PARTY)
  // protobuf::Any is unavailable with third_party/protobuf:protobuf-lite.
#else
  for (const ::mediapipe::protobuf::Any& options : node_config.node_options()) {
    if (options.Is<T>()) {
      options.UnpackTo(result);
    }
  }
#endif
}

// Combine a base options message with an optional side packet. The specified
// packet can hold either the specified options type T or CalculatorOptions.
// Fields are either replaced or merged depending on field merge_fields.
template <typename T>
inline T RetrieveOptions(const T& base, const PacketSet& packet_set,
                         const std::string& tag_name) {
  if (packet_set.HasTag(tag_name)) {
    const Packet& packet = packet_set.Tag(tag_name);
    T packet_options;
    if (packet.ValidateAsType<T>().ok()) {
      packet_options = packet.Get<T>();
    } else if (packet.ValidateAsType<CalculatorOptions>().ok()) {
      GetExtension<T>(packet.Get<CalculatorOptions>(), &packet_options);
    }
    return tool::MergeOptions(base, packet_options);
  }
  return base;
}

// Extracts the options message of a specified type from a
// CalculatorGraphConfig::Node.
class OptionsMap {
 public:
  OptionsMap& Initialize(const CalculatorGraphConfig::Node& node_config) {
    node_config_ = &node_config;
    return *this;
  }

  // Returns the options data for a CalculatorGraphConfig::Node, from
  // either "options" or "node_options" using either GetExtension or UnpackTo.
  template <class T>
  const T& Get() const {
    if (options_.Has<T>()) {
      return *options_.Get<T>();
    }
    T* result = options_.Get<T>();
    if (node_config_->has_options()) {
      GetExtension(node_config_->options(), result);
    } else {
      GetNodeOptions(*node_config_, result);
    }
    return *result;
  }

  const CalculatorGraphConfig::Node* node_config_;
  TypeMap options_;
};

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_UTIL_H_
