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

#include "mediapipe/framework/deps/registration_token.h"

#include <utility>

namespace mediapipe {
RegistrationToken::RegistrationToken(std::function<void()> unregisterer)
    : unregister_function_(std::move(unregisterer)) {}

RegistrationToken::RegistrationToken(RegistrationToken&& rhs)
    : unregister_function_(std::move(rhs.unregister_function_)) {
  rhs.unregister_function_ = nullptr;
}

RegistrationToken& RegistrationToken::operator=(RegistrationToken&& rhs) {
  if (&rhs != this) {
    unregister_function_ = std::move(rhs.unregister_function_);
    rhs.unregister_function_ = nullptr;
  }
  return *this;
}

void RegistrationToken::Unregister() {
  if (unregister_function_ != nullptr) {
    unregister_function_();
    unregister_function_ = nullptr;
  }
}

namespace {
struct CombinedToken {
  void operator()() {
    for (auto& f : functions) {
      f();
    }
  }
  std::vector<std::function<void()>> functions;
};
}  // anonymous namespace

// static
RegistrationToken RegistrationToken::Combine(
    std::vector<RegistrationToken> tokens) {
  CombinedToken combined;

  // When vector grows, it only moves elements if the move constructor is marked
  // noexcept (or if the element isn't copyable). In related news, function's
  // move constructor is not marked noexcept. By reserving the correct amount of
  // space up front, we remove the need for the vector to grow, and thus
  // eliminate copies.
  combined.functions.reserve(tokens.size());
  for (RegistrationToken& token : tokens) {
    combined.functions.push_back(std::move(token.unregister_function_));
  }
  return RegistrationToken(std::move(combined));
}

Unregister::Unregister(RegistrationToken token) : token_(std::move(token)) {}

Unregister::~Unregister() { token_.Unregister(); }

Unregister::Unregister(Unregister&& rhs) : token_(std::move(rhs.token_)) {}
Unregister& Unregister::operator=(Unregister&& rhs) {
  if (&rhs != this) {
    token_.Unregister();
    token_ = std::move(rhs.token_);
  }
  return *this;
}

void Unregister::Reset(RegistrationToken token) {
  token_.Unregister();
  token_ = std::move(token);
}

}  // namespace mediapipe
