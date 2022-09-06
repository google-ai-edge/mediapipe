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

#ifndef MEDIAPIPE_DEPS_REGISTRATION_H_
#define MEDIAPIPE_DEPS_REGISTRATION_H_

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/meta/type_traits.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/registration_token.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

// Usage:
//
// === Defining a registry ================================================
//
//  class Widget {};
//
//  using WidgetRegistry =
//      GlobalFactoryRegistry<unique_ptr<Widget>,                 // return
//                            unique_ptr<Gadget>, const Thing*>   // args
//
// === Registering an implementation =======================================
//
//  class MyWidget : public Widget {
//    static unique_ptr<Widget> Create(unique_ptr<Gadget> arg,
//                                     const Thing* thing) {
//      return MakeUnique<Widget>(std::move(arg), thing);
//    }
//    ...
//  };
//
//  REGISTER_FACTORY_FUNCTION_QUALIFIED(
//      WidgetRegistry, widget_registration,
//      ::my_ns::MyWidget, MyWidget::Create);
//
// === Using std::function =================================================
//
//  class Client {};
//
//  using ClientRegistry =
//      GlobalFactoryRegistry<absl::StatusOr<unique_ptr<Client>>;
//
//  class MyClient : public Client {
//   public:
//    MyClient(unique_ptr<Backend> backend)
//      : backend_(std::move(backend)) {}
//   private:
//     const std::unique_ptr<Backend> backend_;
//  };
//
//  // Any std::function that returns a Client is valid to pass here. Below,
//  // we use a lambda.
//  REGISTER_FACTORY_FUNCTION_QUALIFIED(
//      ClientRegistry, client_registration,
//      ::my_ns::MyClient,
//      []() {
//        auto backend = absl::make_unique<Backend>("/path/to/backend");
//        const absl::Status status = backend->Init();
//        if (!status.ok()) {
//          return status;
//        }
//        std::unique_ptr<Client> client
//            = absl::make_unique<MyClient>(std::move(backend));
//        return client;
//      });
//
// === Using the registry to create instances ==============================
//
//  // Registry will return absl::StatusOr<Object>
//  absl::StatusOr<unique_ptr<Widget>> s_or_widget =
//      WidgetRegistry::CreateByName(
//          "my_ns.MyWidget", std::move(gadget), thing);
//  // Registry will return NOT_FOUND if the name is unknown.
//  if (!s_or_widget.ok()) ... // handle error
//  DoStuffWithWidget(std::move(s_or_widget).value());
//
//  // It's also possible to find an instance by name within a source namespace.
//  auto s_or_widget = WidgetRegistry::CreateByNameInNamespace(
//      "my_ns.sub_namespace", "MyWidget");
//
//  // It's also possible to just check if a name is registered without creating
//  // an instance.
//  bool registered = WidgetRegistry::IsRegistered("my_ns::MyWidget");
//
//  // It's also possible to iterate through all registered function names.
//  // This might be useful if clients outside of your codebase are registering
//  // plugins.
//  for (const auto& name : WidgetRegistry::GetRegisteredNames()) {
//    absl::StatusOr<unique_ptr<Widget>> s_or_widget =
//        WidgetRegistry::CreateByName(name, std::move(gadget), thing);
//    ...
//  }
//
// === Injecting instances for testing =====================================
//
// Unregister unregisterer(WidgetRegistry::Register(
//     "MockWidget",
//      [](unique_ptr<Gadget> arg, const Thing* thing) {
//        ...
//      }));

namespace registration_internal {
inline constexpr char kCxxSep[] = "::";
inline constexpr char kNameSep[] = ".";

template <typename T>
struct WrapStatusOr {
  using type = absl::StatusOr<T>;
};

// Specialization to avoid double-wrapping types that are already StatusOrs.
template <typename T>
struct WrapStatusOr<absl::StatusOr<T>> {
  using type = absl::StatusOr<T>;
};
}  // namespace registration_internal

class NamespaceAllowlist {
 public:
  static const absl::flat_hash_set<std::string>& TopNamespaces();
};

template <typename R, typename... Args>
class FunctionRegistry {
 public:
  using Function = std::function<R(Args...)>;
  using ReturnType = typename registration_internal::WrapStatusOr<R>::type;

  FunctionRegistry() {}
  FunctionRegistry(const FunctionRegistry&) = delete;
  FunctionRegistry& operator=(const FunctionRegistry&) = delete;

  RegistrationToken Register(const std::string& name, Function func)
      ABSL_LOCKS_EXCLUDED(lock_) {
    std::string normalized_name = GetNormalizedName(name);
    absl::WriterMutexLock lock(&lock_);
    std::string adjusted_name = GetAdjustedName(normalized_name);
    if (adjusted_name != normalized_name) {
      functions_.insert(std::make_pair(adjusted_name, func));
    }
    if (functions_.insert(std::make_pair(normalized_name, std::move(func)))
            .second) {
      return RegistrationToken(
          [this, normalized_name]() { Unregister(normalized_name); });
    }
    LOG(FATAL) << "Function with name " << name << " already registered.";
    return RegistrationToken([]() {});
  }

  // Force 'args' to be deduced by templating the function, instead of just
  // accepting Args. This is necessary to make 'args' a forwarding reference as
  // opposed to a plain rvalue reference.
  // https://isocpp.org/blog/2012/11/universal-references-in-c11-scott-meyers
  //
  // The absl::enable_if_t is used to disable this method if Args2 are not
  // convertible to Args. This will allow the compiler to identify the offending
  // line (i.e. the line where the method is called) in the first error message,
  // rather than nesting it multiple levels down the error stack.
  template <typename... Args2,
            absl::enable_if_t<std::is_convertible<std::tuple<Args2...>,
                                                  std::tuple<Args...>>::value,
                              int> = 0>
  ReturnType Invoke(const std::string& name, Args2&&... args)
      ABSL_LOCKS_EXCLUDED(lock_) {
    Function function;
    {
      absl::ReaderMutexLock lock(&lock_);
      auto it = functions_.find(name);
      if (it == functions_.end()) {
        return absl::NotFoundError("No registered object with name: " + name);
      }
      function = it->second;
    }
    return function(std::forward<Args2>(args)...);
  }

  // Invokes the specified factory function and returns the result.
  // Namespaces in |name| and |ns| are separated by kNameSep.
  template <typename... Args2>
  ReturnType Invoke(const std::string& ns, const std::string& name,
                    Args2&&... args) ABSL_LOCKS_EXCLUDED(lock_) {
    return Invoke(GetQualifiedName(ns, name), args...);
  }

  // Note that it's possible for registered implementations to be subsequently
  // unregistered, though this will never happen with registrations made via
  // MEDIAPIPE_REGISTER_FACTORY_FUNCTION.
  bool IsRegistered(const std::string& name) const ABSL_LOCKS_EXCLUDED(lock_) {
    absl::ReaderMutexLock lock(&lock_);
    return functions_.count(name) != 0;
  }

  // Returns true if the specified factory function is available.
  // Namespaces in |name| and |ns| are separated by kNameSep.
  bool IsRegistered(const std::string& ns, const std::string& name) const
      ABSL_LOCKS_EXCLUDED(lock_) {
    return IsRegistered(GetQualifiedName(ns, name));
  }

  // Returns a vector of all registered function names.
  // Note that it's possible for registered implementations to be subsequently
  // unregistered, though this will never happen with registrations made via
  // MEDIAPIPE_REGISTER_FACTORY_FUNCTION.
  std::unordered_set<std::string> GetRegisteredNames() const
      ABSL_LOCKS_EXCLUDED(lock_) {
    absl::ReaderMutexLock lock(&lock_);
    std::unordered_set<std::string> names;
    std::for_each(functions_.cbegin(), functions_.cend(),
                  [&names](const std::pair<const std::string, Function>& pair) {
                    names.insert(pair.first);
                  });
    return names;
  }

  // Normalizes a C++ qualified name.  Validates the name qualification.
  // The name must be either unqualified or fully qualified with a leading "::".
  // The leading "::" in a fully qualified name is stripped.
  std::string GetNormalizedName(const std::string& name) {
    using ::mediapipe::registration_internal::kCxxSep;
    std::vector<std::string> names = absl::StrSplit(name, kCxxSep);
    if (names[0].empty()) {
      names.erase(names.begin());
    } else {
      CHECK_EQ(1, names.size())
          << "A registered class name must be either fully qualified "
          << "with a leading :: or unqualified, got: " << name << ".";
    }
    return absl::StrJoin(names, kCxxSep);
  }

  // Returns the registry key for a name specified within a namespace.
  // Namespaces are separated by kNameSep.
  std::string GetQualifiedName(const std::string& ns,
                               const std::string& name) const {
    using ::mediapipe::registration_internal::kCxxSep;
    using ::mediapipe::registration_internal::kNameSep;
    std::vector<std::string> names = absl::StrSplit(name, kNameSep);
    if (names[0].empty()) {
      names.erase(names.begin());
      return absl::StrJoin(names, kCxxSep);
    }
    std::string cxx_name = absl::StrJoin(names, kCxxSep);
    if (ns.empty()) {
      return cxx_name;
    }
    std::vector<std::string> spaces = absl::StrSplit(ns, kNameSep);
    absl::ReaderMutexLock lock(&lock_);
    while (!spaces.empty()) {
      std::string cxx_ns = absl::StrJoin(spaces, kCxxSep);
      std::string qualified_name = absl::StrCat(cxx_ns, kCxxSep, cxx_name);
      if (functions_.count(qualified_name)) {
        return qualified_name;
      }
      spaces.pop_back();
    }
    return cxx_name;
  }

 private:
  mutable absl::Mutex lock_;
  std::unordered_map<std::string, Function> functions_ ABSL_GUARDED_BY(lock_);

  // For names included in NamespaceAllowlist, strips the namespace.
  std::string GetAdjustedName(const std::string& name) {
    using ::mediapipe::registration_internal::kCxxSep;
    std::vector<std::string> names = absl::StrSplit(name, kCxxSep);
    std::string base_name = names.back();
    names.pop_back();
    std::string ns = absl::StrJoin(names, kCxxSep);
    if (NamespaceAllowlist::TopNamespaces().count(ns)) {
      return base_name;
    }
    return name;
  }

  void Unregister(const std::string& name) {
    absl::WriterMutexLock lock(&lock_);
    std::string adjusted_name = GetAdjustedName(name);
    if (adjusted_name != name) {
      functions_.erase(adjusted_name);
    }
    functions_.erase(name);
  }
};

template <typename R, typename... Args>
class GlobalFactoryRegistry {
  using Functions = FunctionRegistry<R, Args...>;

 public:
  static RegistrationToken Register(const std::string& name,
                                    typename Functions::Function func) {
    return functions()->Register(name, std::move(func));
  }

  // Invokes the specified factory function and returns the result.
  // If using namespaces with this registry, the variant with a namespace
  // argument should be used.
  template <typename... Args2>
  static typename Functions::ReturnType CreateByName(const std::string& name,
                                                     Args2&&... args) {
    return functions()->Invoke(name, std::forward<Args2>(args)...);
  }

  // Returns true if the specified factory function is available.
  // If using namespaces with this registry, the variant with a namespace
  // argument should be used.
  static bool IsRegistered(const std::string& name) {
    return functions()->IsRegistered(name);
  }

  static std::unordered_set<std::string> GetRegisteredNames() {
    return functions()->GetRegisteredNames();
  }

  // Invokes the specified factory function and returns the result.
  // Namespaces in |name| and |ns| are separated by kNameSep.
  // See comments re: use of Args2 and absl::enable_if_t on Invoke.
  template <typename... Args2,
            absl::enable_if_t<std::is_convertible<std::tuple<Args2...>,
                                                  std::tuple<Args...>>::value,
                              int> = 0>
  static typename Functions::ReturnType CreateByNameInNamespace(
      const std::string& ns, const std::string& name, Args2&&... args) {
    return functions()->Invoke(ns, name, std::forward<Args2>(args)...);
  }

  // Returns true if the specified factory function is available.
  // Namespaces in |name| and |ns| are separated by kNameSep.
  static bool IsRegistered(const std::string& ns, const std::string& name) {
    return functions()->IsRegistered(ns, name);
  }

  // Returns the factory function registry singleton.
  static Functions* functions() {
    static auto* functions = new Functions();
    return functions;
  }

 private:
  GlobalFactoryRegistry() = delete;
};

// Two levels of macros are required to convert __LINE__ into a string
// containing the line number.
#define REGISTRY_STATIC_VAR_INNER(var_name, line) var_name##_##line##__
#define REGISTRY_STATIC_VAR(var_name, line) \
  REGISTRY_STATIC_VAR_INNER(var_name, line)

#define MEDIAPIPE_REGISTER_FACTORY_FUNCTION(RegistryType, name, ...) \
  static auto* REGISTRY_STATIC_VAR(registration_##name, __LINE__) =  \
      new mediapipe::RegistrationToken(                              \
          RegistryType::Register(#name, __VA_ARGS__))

#define REGISTER_FACTORY_FUNCTION_QUALIFIED(RegistryType, var_name, name, ...) \
  static auto* REGISTRY_STATIC_VAR(var_name, __LINE__) =                       \
      new mediapipe::RegistrationToken(                                        \
          RegistryType::Register(#name, __VA_ARGS__))

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_REGISTRATION_H_
