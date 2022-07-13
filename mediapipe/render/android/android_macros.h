// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// This file contains macros and macro-like constructs (e.g., templates) that
// are commonly used throughout Chromium source. (It may also contain things
// that are closely related to things that are commonly used that belong in this
// file.)
#ifndef BASE_MACROS_H_
#define BASE_MACROS_H_
// ALL DISALLOW_xxx MACROS ARE DEPRECATED; DO NOT USE IN NEW CODE.
// Use explicit deletions instead.  See the section on copyability/movability in
// //styleguide/c++/c++-dos-and-donts.md for more information.
// DEPRECATED: See above. Makes a class uncopyable.
#define DISALLOW_COPY(TypeName) \
  TypeName(const TypeName&) = delete
// DEPRECATED: See above. Makes a class unassignable.
#define DISALLOW_ASSIGN(TypeName) TypeName& operator=(const TypeName&) = delete
// DEPRECATED: See above. Makes a class uncopyable and unassignable.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName);                 \
  DISALLOW_ASSIGN(TypeName)
// DEPRECATED: See above. Disallow all implicit constructors, namely the
// default constructor, copy constructor and operator= functions.
#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName) \
  TypeName() = delete;                           \
  DISALLOW_COPY_AND_ASSIGN(TypeName)
// Used to explicitly mark the return value of a function as unused. If you are
// really sure you don't want to do anything with the return value of a function
// that has been marked WARN_UNUSED_RESULT, wrap it with this. Example:
//
//   std::unique_ptr<MyType> my_var = ...;
//   if (TakeOwnership(my_var.get()) == SUCCESS)
//     ignore_result(my_var.release());
//
template<typename T>
inline void ignore_result(const T&) {
}
#endif  // BASE_MACROS_H_