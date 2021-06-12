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

#include <functional>
#include <utility>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {
class RegistrationTokenTest : public testing::Test {
 public:
  void CallFirst() { ++called_1_; }

  void CallSecond() { ++called_2_; }

  void CallThird() { ++called_3_; }

 protected:
  int called_1_{0};
  int called_2_{0};
  int called_3_{0};
};

// Trivial unregistration test.
TEST_F(RegistrationTokenTest, TestUnregister) {
  std::function<void()> caller = [this]() {
    RegistrationTokenTest::CallFirst();
  };
  RegistrationToken token(caller);
  ASSERT_EQ(0, called_1_);
  token.Unregister();
  ASSERT_EQ(1, called_1_);

  // Check that further calls have no effect.
  token.Unregister();
  token.Unregister();
  ASSERT_EQ(1, called_1_);

  // Test the RAII class.
  ASSERT_EQ(0, called_2_);
  RegistrationToken token2([this]() { RegistrationTokenTest::CallSecond(); });
  {
    Unregister t(std::move(token2));
    ASSERT_EQ(0, called_2_);
  }

  // It was called since the Unregister() went out of scope.
  ASSERT_EQ(1, called_2_);
}

// Tests that the result of a Combine() token does unregisters all combined
// tokens.
TEST_F(RegistrationTokenTest, TestCombine) {
  std::function<void()> caller_1 = [this]() {
    RegistrationTokenTest::CallFirst();
  };
  std::function<void()> caller_2 = [this]() {
    RegistrationTokenTest::CallSecond();
  };
  std::function<void()> caller_3 = [this]() {
    RegistrationTokenTest::CallThird();
  };

  RegistrationToken token_1(caller_1);
  RegistrationToken token_2(caller_2);
  RegistrationToken token_3(caller_3);

  ASSERT_EQ(0, called_1_);
  ASSERT_EQ(0, called_2_);
  ASSERT_EQ(0, called_3_);

  std::vector<RegistrationToken> tokens;
  tokens.emplace_back(std::move(token_1));
  tokens.emplace_back(std::move(token_2));
  tokens.emplace_back(std::move(token_3));

  RegistrationToken combined = RegistrationToken::Combine(std::move(tokens));
  combined.Unregister();

  ASSERT_EQ(1, called_1_);
  ASSERT_EQ(1, called_2_);
  ASSERT_EQ(1, called_3_);

  // Check that the original tokens were invalidated by their move and do
  // nothing.
  token_1.Unregister();
  token_2.Unregister();
  token_3.Unregister();

  ASSERT_EQ(1, called_1_);
  ASSERT_EQ(1, called_2_);
  ASSERT_EQ(1, called_3_);
}

TEST_F(RegistrationTokenTest, TestMove) {
  RegistrationToken token([this] { CallFirst(); });
  token = RegistrationToken([this] { CallFirst(); });
  EXPECT_EQ(0, called_1_);

  Unregister unreg;
  unreg = Unregister(std::move(token));
  EXPECT_EQ(0, called_1_);
  unreg = Unregister(RegistrationToken([this] { CallFirst(); }));
  EXPECT_EQ(1, called_1_);
  unreg = Unregister(RegistrationToken([this] { CallFirst(); }));
  EXPECT_EQ(2, called_1_);
  unreg.Reset();
  EXPECT_EQ(3, called_1_);
}

}  // namespace
}  // namespace mediapipe
