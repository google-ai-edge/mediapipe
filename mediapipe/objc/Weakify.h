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
//
// OBJC_LINTER

// TODO: check license. This was forked from an internal header.

/**
 * __WEAKNAME_ is a private macro used to generate a local variable name related
 * to the argument variable name. This generated local variable name is
 * intentionally stable across multiple invocations.
 */
#define __WEAKNAME_(variable) variable##_weak_

/**
 * WEAKIFY defines a new local variable that is a weak reference to the argument
 * variable.
 *
 * This macro is generally used to capture a weak reference to be captured by an
 * Objective-C block to avoid unintentionally extending an object's lifetime or
 * avoid causing a retain cycle.
 *
 * The new local variable's name will be based on the name of the target
 * variable and is stable across multiple invocations of WEAKIFY. In general,
 * you should not need to invoke WEAKIFY multiple times on the same variable.
 */
#define WEAKIFY(variable) \
  __weak __typeof__(variable) __WEAKNAME_(variable) = (variable)

/**
 * STRONGIFY defines a new shadow local variable with the same name as the
 * argument variable and initialize it with a resolved weak reference based on a
 * weak reference created previously using the WEAKIFY macro.
 *
 * @note:
 * This macro must be called within each block scope to prevent nested blocks
 * from capturing a strong reference from an outer block.
 */
#define STRONGIFY(variable)                            \
  _Pragma("clang diagnostic push")                     \
      _Pragma("clang diagnostic ignored \"-Wshadow\"") \
          __strong __typeof__(variable) variable =     \
              __WEAKNAME_(variable) _Pragma("clang diagnostic pop")
