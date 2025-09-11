#ifndef MEDIAPIPE_DEPS_COMPILE_TIME_STRING_H_
#define MEDIAPIPE_DEPS_COMPILE_TIME_STRING_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>

#include "absl/strings/string_view.h"

namespace mediapipe {
namespace internal {

// Calling these functions is not a manifestly constant-evaluation, which we use
// below to cause compilation to fail when certain conditions are not met.
inline void CharArrayInputMustNotContainEmbeddedNULs() { std::abort(); }
inline void CharArrayInputMustBeNULTerminated() { std::abort(); }

}  // namespace internal

// An immutable string that is guaranteed to be defined at compile-time.
// Implemented as a structural type so it may be used as a template parameter:
// https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter
// May be initialized from a string literal, string_view constant, or
// NUL-terminated char array constant. Example usage:
//   template <CompileTimeString counter_path>
//   void IncrementCounter() { ... }
//
//   IncrementCounter<"/path/to/counter">();
template <size_t storage_size>
class CompileTimeString final {
 public:
  // Constructs the string from a NUL-terminated string literal or char array.
  // Example:
  //   CompileTimeString("hello world")
  // Note: the string must contain exactly one NUL character as the final
  // character which is dropped from the input and is not included in the size
  // of the CompileTimeString.
  template <size_t length>
  consteval CompileTimeString(  // NOLINT: implicit conversions intended.
      const char (&s)[length])
      : internal_stored_str_(AsStdArray(s)), internal_view_(kEmptyStringView) {}

  // Constructs the string from a string_view constant. Example:
  //   static constexpr absl::string_view kFoo = "foo";
  //   CompileTimeString(kFoo)
  consteval CompileTimeString(     // NOLINT: implicit conversions intended.
      const absl::string_view& v)  // NOLINT: need ref for structural type.
      : internal_stored_str_(kEmptyStoredStr), internal_view_(v) {}

  // Does not support copy or assign.
  CompileTimeString(const CompileTimeString&) = delete;
  CompileTimeString& operator=(const CompileTimeString&) = delete;

  constexpr size_t size() const {
    if constexpr (storage_size == 0) {
      return internal_view_.size();
    }
    return storage_size;
  }
  constexpr size_t length() const { return size(); }

  constexpr bool empty() const { return size() == 0; }

  // Returns a string_view referencing a string with lifetime equal to the
  // lifetime of the CompileTimeString.
  constexpr absl::string_view AsStringView() const {
    if constexpr (storage_size == 0) {
      return internal_view_;
    } else {
      return absl::string_view(internal_stored_str_.data(), storage_size);
    }
  }

  // Logically internal but must be public to be a structural type.
 public:
  // The data when constructed from string literal or char array.
  // `kEmptyStoredStr` when constructed from string_view.
  // NOLINTNEXTLINE: logically internal.
  const std::array<char, storage_size> internal_stored_str_;
  // The data when constructed from string_view. `kEmptyStringView` when
  // constructed from string literal or char array. Must be a reference since
  // string_view is not a structural type.
  const absl::string_view& internal_view_;  // NOLINT: logically internal.

 private:
  // Empty value for `view` when constructed from non-string_view.
  static constexpr absl::string_view kEmptyStringView = "";
  // Empty value for `internal_stored_str_` when constructed from string_view,
  // "", or character array containing only NUL.
  static constexpr std::array<char, 0> kEmptyStoredStr = {};

  // Creates a std::array of characters from a literal string or char array,
  // discarding the terminal NUL character. Assumes `length` = `storage_size`+1
  // per deduction guide below.
  template <size_t length>
  static consteval std::array<char, storage_size> AsStdArray(
      const char (&s)[length]) {
    ValidateCharArray(s);  // Won't compile if `s` isn't valid.

    if constexpr (length == 1) {
      // Small optimization if the string is a single NUL character.
      return kEmptyStoredStr;
    } else {
      std::array<char, storage_size> out;
      // Copy everything except the NUL character.
      std::copy_n(s, storage_size, out.data());
      return out;
    }
  }

  // Fails to compile if `s` contains embedded NULs or does not end with NUL.
  template <size_t length>
  static consteval void ValidateCharArray(const char (&s)[length]) {
    for (size_t i = 0; i < storage_size; ++i) {
      if (s[i] == '\0') {
        internal::CharArrayInputMustNotContainEmbeddedNULs();
      }
    }
    if (s[storage_size] != '\0') {
      internal::CharArrayInputMustBeNULTerminated();
    }
  }
};

template <size_t length>
CompileTimeString(const char (&)[length]) -> CompileTimeString<length - 1>;
CompileTimeString(const absl::string_view&) -> CompileTimeString<0>;

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_COMPILE_TIME_STRING_H_
