#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_INTERNAL_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_INTERNAL_H_

#include <cstdint>

namespace mediapipe {

// Generates unique view id at compile-time using FILE and LINE.
#define TENSOR_UNIQUE_VIEW_ID()                               \
  static constexpr uint64_t kId = tensor_internal::FnvHash64( \
      __FILE__, tensor_internal::FnvHash64(TENSOR_INT_TO_STRING(__LINE__)))

namespace tensor_internal {

#define TENSOR_INT_TO_STRING2(x) #x
#define TENSOR_INT_TO_STRING(x) TENSOR_INT_TO_STRING2(x)

// Compile-time hash function
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
constexpr uint64_t kFnvPrime = 0x00000100000001B3;
constexpr uint64_t kFnvOffsetBias = 0xcbf29ce484222325;
constexpr uint64_t FnvHash64(const char* str, uint64_t hash = kFnvOffsetBias) {
  return (str[0] == 0) ? hash : FnvHash64(str + 1, (hash ^ str[0]) * kFnvPrime);
}
}  // namespace tensor_internal
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_INTERNAL_H_
