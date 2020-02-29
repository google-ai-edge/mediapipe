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

// Unit tests for StatusOr

#include "mediapipe/framework/deps/statusor.h"

#include <memory>
#include <type_traits>

#include "mediapipe/framework/deps/canonical_errors.h"
#include "mediapipe/framework/deps/status.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

class Base1 {
 public:
  virtual ~Base1() {}
  int pad_;
};

class Base2 {
 public:
  virtual ~Base2() {}
  int yetotherpad_;
};

class Derived : public Base1, public Base2 {
 public:
  ~Derived() override {}
  int evenmorepad_;
};

class CopyNoAssign {
 public:
  explicit CopyNoAssign(int value) : foo_(value) {}
  CopyNoAssign(const CopyNoAssign& other) : foo_(other.foo_) {}
  int foo_;

 private:
  const CopyNoAssign& operator=(const CopyNoAssign&);
};

class NoDefaultConstructor {
 public:
  explicit NoDefaultConstructor(int foo);
};

static_assert(!std::is_default_constructible<NoDefaultConstructor>(),
              "Should not be default-constructible.");

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return std::unique_ptr<int>(new int(0));
}

TEST(StatusOr, ElementType) {
  static_assert(std::is_same<StatusOr<int>::element_type, int>(), "");
  static_assert(std::is_same<StatusOr<char>::element_type, char>(), "");
}

TEST(StatusOr, TestNoDefaultConstructorInitialization) {
  // Explicitly initialize it with an error code.
  ::mediapipe::StatusOr<NoDefaultConstructor> statusor(
      ::mediapipe::CancelledError(""));
  EXPECT_FALSE(statusor.ok());
  EXPECT_EQ(statusor.status().code(), ::mediapipe::StatusCode::kCancelled);

  // Default construction of StatusOr initializes it with an UNKNOWN error code.
  ::mediapipe::StatusOr<NoDefaultConstructor> statusor2;
  EXPECT_FALSE(statusor2.ok());
  EXPECT_EQ(statusor2.status().code(), ::mediapipe::StatusCode::kUnknown);
}

TEST(StatusOr, TestMoveOnlyInitialization) {
  ::mediapipe::StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  int* previous = thing.ValueOrDie().get();

  thing = ReturnUniquePtr();
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  EXPECT_NE(previous, thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyStatusCtr) {
  ::mediapipe::StatusOr<std::unique_ptr<int>> thing(
      ::mediapipe::CancelledError(""));
  ASSERT_FALSE(thing.ok());
}

TEST(StatusOr, TestMoveOnlyValueExtraction) {
  ::mediapipe::StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  std::unique_ptr<int> ptr = thing.ConsumeValueOrDie();
  EXPECT_EQ(0, *ptr);

  thing = std::move(ptr);
  ptr = std::move(thing.ValueOrDie());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestMoveOnlyConversion) {
  ::mediapipe::StatusOr<std::unique_ptr<const int>> const_thing(
      ReturnUniquePtr());
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());

  // Test rvalue converting assignment
  const int* const_previous = const_thing.ValueOrDie().get();
  const_thing = ReturnUniquePtr();
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());
  EXPECT_NE(const_previous, const_thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyVector) {
  // Sanity check that ::mediapipe::StatusOr<MoveOnly> works in vector.
  std::vector<::mediapipe::StatusOr<std::unique_ptr<int>>> vec;
  vec.push_back(ReturnUniquePtr());
  vec.resize(2);
  auto another_vec = std::move(vec);
  EXPECT_EQ(0, *another_vec[0].ValueOrDie());
  EXPECT_EQ(::mediapipe::StatusCode::kUnknown, another_vec[1].status().code());
}

TEST(StatusOr, TestMoveWithValuesAndErrors) {
  ::mediapipe::StatusOr<std::string> status_or(std::string(1000, '0'));
  ::mediapipe::StatusOr<std::string> value1(std::string(1000, '1'));
  ::mediapipe::StatusOr<std::string> value2(std::string(1000, '2'));
  ::mediapipe::StatusOr<std::string> error1(
      Status(::mediapipe::StatusCode::kUnknown, "error1"));
  ::mediapipe::StatusOr<std::string> error2(
      Status(::mediapipe::StatusCode::kUnknown, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.ValueOrDie());

  // Overwrite the value in status_or with another value.
  status_or = std::move(value1);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.ValueOrDie());

  // Overwrite the value in status_or with an error.
  status_or = std::move(error1);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().message());

  // Overwrite the error in status_or with another error.
  status_or = std::move(error2);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().message());

  // Overwrite the error with a value.
  status_or = std::move(value2);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.ValueOrDie());
}

TEST(StatusOr, TestCopyWithValuesAndErrors) {
  ::mediapipe::StatusOr<std::string> status_or(std::string(1000, '0'));
  ::mediapipe::StatusOr<std::string> value1(std::string(1000, '1'));
  ::mediapipe::StatusOr<std::string> value2(std::string(1000, '2'));
  ::mediapipe::StatusOr<std::string> error1(
      Status(::mediapipe::StatusCode::kUnknown, "error1"));
  ::mediapipe::StatusOr<std::string> error2(
      Status(::mediapipe::StatusCode::kUnknown, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.ValueOrDie());

  // Overwrite the value in status_or with another value.
  status_or = value1;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.ValueOrDie());

  // Overwrite the value in status_or with an error.
  status_or = error1;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().message());

  // Overwrite the error in status_or with another error.
  status_or = error2;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().message());

  // Overwrite the error with a value.
  status_or = value2;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.ValueOrDie());

  // Verify original values unchanged.
  EXPECT_EQ(std::string(1000, '1'), value1.ValueOrDie());
  EXPECT_EQ("error1", error1.status().message());
  EXPECT_EQ("error2", error2.status().message());
  EXPECT_EQ(std::string(1000, '2'), value2.ValueOrDie());
}

TEST(StatusOr, TestDefaultCtor) {
  ::mediapipe::StatusOr<int> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), ::mediapipe::StatusCode::kUnknown);
}

TEST(StatusOrDeathTest, TestDefaultCtorValue) {
  ::mediapipe::StatusOr<int> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "");

  const ::mediapipe::StatusOr<int> thing2;
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestStatusCtor) {
  ::mediapipe::StatusOr<int> thing(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), ::mediapipe::StatusCode::kCancelled);
}

TEST(StatusOr, TestValueCtor) {
  const int kI = 4;
  const ::mediapipe::StatusOr<int> thing(kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusOk) {
  const int kI = 4;
  const ::mediapipe::StatusOr<int> original(kI);
  const ::mediapipe::StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOk) {
  ::mediapipe::StatusOr<int> original(
      Status(::mediapipe::StatusCode::kCancelled, ""));
  ::mediapipe::StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestCopyCtorNonAssignable) {
  const int kI = 4;
  CopyNoAssign value(kI);
  ::mediapipe::StatusOr<CopyNoAssign> original(value);
  ::mediapipe::StatusOr<CopyNoAssign> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie().foo_, copy.ValueOrDie().foo_);
}

TEST(StatusOr, TestCopyCtorStatusOKConverting) {
  const int kI = 4;
  ::mediapipe::StatusOr<int> original(kI);
  ::mediapipe::StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_DOUBLE_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOkConverting) {
  ::mediapipe::StatusOr<int> original(
      Status(::mediapipe::StatusCode::kCancelled, ""));
  ::mediapipe::StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestAssignmentStatusOk) {
  const int kI = 4;
  ::mediapipe::StatusOr<int> source(kI);
  ::mediapipe::StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestAssignmentStatusNotOk) {
  ::mediapipe::StatusOr<int> source(
      Status(::mediapipe::StatusCode::kCancelled, ""));
  ::mediapipe::StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestStatus) {
  ::mediapipe::StatusOr<int> good(4);
  EXPECT_TRUE(good.ok());
  ::mediapipe::StatusOr<int> bad(
      Status(::mediapipe::StatusCode::kCancelled, ""));
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.status(), Status(::mediapipe::StatusCode::kCancelled, ""));
}

TEST(StatusOr, TestValue) {
  const int kI = 4;
  ::mediapipe::StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestValueConst) {
  const int kI = 4;
  const ::mediapipe::StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOrDeathTest, TestValueNotOk) {
  ::mediapipe::StatusOr<int> thing(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

TEST(StatusOrDeathTest, TestValueNotOkConst) {
  const ::mediapipe::StatusOr<int> thing(
      ::mediapipe::Status(::mediapipe::StatusCode::kUnknown, ""));
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestPointerDefaultCtor) {
  ::mediapipe::StatusOr<int*> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), ::mediapipe::StatusCode::kUnknown);
}

TEST(StatusOrDeathTest, TestPointerDefaultCtorValue) {
  ::mediapipe::StatusOr<int*> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestPointerStatusCtor) {
  ::mediapipe::StatusOr<int*> thing(
      Status(::mediapipe::StatusCode::kCancelled, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status(), Status(::mediapipe::StatusCode::kCancelled, ""));
}

TEST(StatusOr, TestPointerValueCtor) {
  const int kI = 4;
  ::mediapipe::StatusOr<const int*> thing(&kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusOk) {
  const int kI = 0;
  ::mediapipe::StatusOr<const int*> original(&kI);
  ::mediapipe::StatusOr<const int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOk) {
  ::mediapipe::StatusOr<int*> original(
      Status(::mediapipe::StatusCode::kCancelled, ""));
  ::mediapipe::StatusOr<int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerCopyCtorStatusOKConverting) {
  Derived derived;
  ::mediapipe::StatusOr<Derived*> original(&derived);
  ::mediapipe::StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(static_cast<const Base2*>(original.ValueOrDie()),
            copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOkConverting) {
  ::mediapipe::StatusOr<Derived*> original(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, ""));
  ::mediapipe::StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerAssignmentStatusOk) {
  const int kI = 0;
  ::mediapipe::StatusOr<const int*> source(&kI);
  ::mediapipe::StatusOr<const int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOk) {
  ::mediapipe::StatusOr<int*> source(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, ""));
  ::mediapipe::StatusOr<int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestPointerStatus) {
  const int kI = 0;
  ::mediapipe::StatusOr<const int*> good(&kI);
  EXPECT_TRUE(good.ok());
  ::mediapipe::StatusOr<const int*> bad(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, ""));
  EXPECT_EQ(bad.status(),
            ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, ""));
}

TEST(StatusOr, TestPointerValue) {
  const int kI = 0;
  ::mediapipe::StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerValueConst) {
  const int kI = 0;
  const ::mediapipe::StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOrDeathTest, TestPointerValueNotOk) {
  ::mediapipe::StatusOr<int*> thing(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

TEST(StatusOrDeathTest, TestPointerValueNotOkConst) {
  const ::mediapipe::StatusOr<int*> thing(
      ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

}  // namespace
}  // namespace mediapipe
