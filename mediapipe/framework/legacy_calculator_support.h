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

#ifndef MEDIAPIPE_FRAMEWORK_LEGACY_CALCULATOR_SUPPORT_H_
#define MEDIAPIPE_FRAMEWORK_LEGACY_CALCULATOR_SUPPORT_H_

#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"

// In Emscripten builds without threading support, some member variables
// declared "static thread_local" will not be linked correctly. To workaround
// this we declare them only "static".
// TODO: remove this macro and use thread_local unconditionally
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
#define EMSCRIPTEN_WORKAROUND_FOR_B121216479 1
#endif

namespace mediapipe {
class GlCalculatorHelper;
class MetalHelperLegacySupport;
namespace aimatter {
template <class T>
class CachableAsyncLoadableObject;
}
}  // namespace mediapipe

namespace xeno {
namespace effect {
class AssetRegistryServiceHelper;
}  // namespace effect
}  // namespace xeno

namespace mediapipe {

class LegacyCalculatorSupport {
 public:
  // Scoped is a RAII helper for setting the current CC in the current scope,
  // and unsetting it automatically (restoring the previous value) when
  // leaving the scope.
  //
  // This allows the current CC to be accessed at any point deeper in the
  // call stack of the current thread, until the scope is left. Creating
  // another Scoped instance deeper in the call stack applies to calls branching
  // from that point, and the previous value is restored when execution leaves
  // that scope, as one would expect.
  //
  // The current value is only accessible via this mechanism by a limited set
  // of classes (listed as friends below). This is only meant to be used where
  // backwards compatibility reasons prevent passing the CC directly.
  //
  // Only two specializations are allowed: Scoped<CalculatorContext> and
  // Scoped<CalculatorContract>.
  template <class C>
  class Scoped {
   public:
    // The constructor saves the current value of current_ in an instance
    // member, which is then restored by the destructor.
    explicit Scoped(C* cc) {
      saved_ = current_;
      current_ = cc;
    }
    ~Scoped() { current_ = saved_; }

   private:
    // The value to restore after exiting this scope.
    C* saved_;

    // The current C* for this thread.
    //
    // This needs NOLINT because, when included in Objective-C++ files,
    // clang-tidy suggests using an Objective-C naming convention, which is
    // inappropriate. (b/116015736) No category specifier because of b/71698089.
#if EMSCRIPTEN_WORKAROUND_FOR_B121216479
    ABSL_CONST_INIT static C* current_;  // NOLINT
#else
    ABSL_CONST_INIT static thread_local C* current_;  // NOLINT
#endif

    static C* current() { return current_; }

    // Only these classes are allowed to access the current CC via this
    // mechanism.
    friend class ::mediapipe::GlCalculatorHelper;
    friend class ::mediapipe::MetalHelperLegacySupport;
    template <class T>
    friend class ::mediapipe::aimatter::CachableAsyncLoadableObject;
    friend class ::xeno::effect::AssetRegistryServiceHelper;
  };
};

// We only declare this variable for two specializations of the template because
// it is only meant to be used for these two types.
#if EMSCRIPTEN_WORKAROUND_FOR_B121216479
template <>
CalculatorContext* LegacyCalculatorSupport::Scoped<CalculatorContext>::current_;
template <>
CalculatorContract*
    LegacyCalculatorSupport::Scoped<CalculatorContract>::current_;
#elif _MSC_VER
// MSVC interprets these declarations as definitions and during linking it
// generates an error about multiple definitions of current_.
#else
template <>
thread_local CalculatorContext*
    LegacyCalculatorSupport::Scoped<CalculatorContext>::current_;
template <>
thread_local CalculatorContract*
    LegacyCalculatorSupport::Scoped<CalculatorContract>::current_;
#endif  // EMSCRIPTEN_WORKAROUND_FOR_B121216479

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_LEGACY_CALCULATOR_SUPPORT_H_
