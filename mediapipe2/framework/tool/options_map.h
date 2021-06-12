#ifndef MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_MAP_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_MAP_H_

#include <map>
#include <memory>
#include <type_traits>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/any_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/type_util.h"

namespace mediapipe {

namespace tool {

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
  for (const mediapipe::protobuf::Any& options : node_config.node_options()) {
    if (options.Is<T>()) {
      options.UnpackTo(result);
    }
  }
#endif
}

// A map from object type to object.
class TypeMap {
 public:
  template <class T>
  bool Has() const {
    return content_.count(TypeId<T>()) > 0;
  }
  template <class T>
  T* Get() const {
    if (!Has<T>()) {
      content_[TypeId<T>()] = std::make_shared<T>();
    }
    return static_cast<T*>(content_[TypeId<T>()].get());
  }

 private:
  mutable std::map<TypeIndex, std::shared_ptr<void>> content_;
};

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

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_MAP_H_
