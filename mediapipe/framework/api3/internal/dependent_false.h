#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_DEPENDENT_FALSE_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_DEPENDENT_FALSE_H_

#include <type_traits>

namespace mediapipe::api3 {

// Workaround for static_assert(false). Example:
//   dependent_false<T>::value returns false.
// For more information, see:
// https://en.cppreference.com/w/cpp/language/if#Constexpr_If
// TODO: migrate to a common utility when available.
template <class T>
struct dependent_false : std::false_type {};

template <class T>
inline constexpr auto dependent_false_v = dependent_false<T>{};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_DEPENDENT_FALSE_H_
