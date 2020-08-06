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

#ifndef MEDIAPIPE_FRAMEWORK_TOOL_TYPE_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_TYPE_UTIL_H_

#include <cstddef>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "mediapipe/framework/port.h"

namespace mediapipe {
namespace tool {

#if !MEDIAPIPE_HAS_RTTI
// A unique identifier for type T.
class TypeInfo {
 public:
  size_t hash_code() const { return reinterpret_cast<size_t>(this); }
  bool operator==(const TypeInfo& other) const { return &other == this; }
  bool operator<(const TypeInfo& other) const { return &other < this; }
  const char* name() const { return "<unknown>"; }
  template <typename T>
  static const TypeInfo& Get() {
    static TypeInfo* static_type_info = new TypeInfo;
    return *static_type_info;
  }

 private:
  TypeInfo() {}
  TypeInfo(const TypeInfo&) = delete;
};

#else  // MEDIAPIPE_HAS_RTTI
// The std unique identifier for type T.
class TypeInfo {
 public:
  size_t hash_code() const { return info_.hash_code(); }
  bool operator==(const TypeInfo& o) const { return info_ == o.info_; }
  bool operator<(const TypeInfo& o) const { return info_.before(o.info_); }
  const char* name() const { return info_.name(); }
  template <typename T>
  static const TypeInfo& Get() {
    static TypeInfo* static_type_info = new TypeInfo(typeid(T));
    return *static_type_info;
  }

 private:
  TypeInfo(const std::type_info& info) : info_(info) {}
  TypeInfo(const TypeInfo&) = delete;

 private:
  const std::type_info& info_;
  friend class TypeIndex;
};
#endif

// An associative key for TypeInfo.
class TypeIndex {
 public:
  TypeIndex(const TypeInfo& info) : info_(info) {}
  size_t hash_code() const { return info_.hash_code(); }
  bool operator==(const TypeIndex& other) const { return info_ == other.info_; }
  bool operator<(const TypeIndex& other) const { return info_ < other.info_; }

 private:
  const TypeInfo& info_;
};

// Returns a unique identifier for type T.
template <typename T>
const TypeInfo& TypeId() {
  return TypeInfo::Get<T>();
}

// Helper method that returns a hash code of the given type. This allows for
// typeid testing across multiple binaries, unlike FastTypeId which used a
// memory location that only works within the same binary. Moreover, we use this
// for supporting multiple .so binaries in a single Android app built using the
// same compiler and C++ libraries.
// Note that std::type_info may still generate the same hash code for different
// types, although the c++ standard recommends that implementations avoid this
// as much as possible.
template <typename T>
size_t GetTypeHash() {
  return TypeId<T>().hash_code();
}

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_TYPE_UTIL_H_
