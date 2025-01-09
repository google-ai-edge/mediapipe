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

// Unit test cases for SafeInt.  Some of this overlaps with the testing for
// StrongInt, but it's important to test not only that SafeInt fails when
// expected, but that it passes when expected.

#include "mediapipe/framework/deps/safe_int.h"

#include <limits>

#include "mediapipe/framework/port/gtest.h"

MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeInt8, int8_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeUInt8, uint8_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeInt16, int16_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeUInt16, uint16_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeInt32, int32_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeInt64, int64_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeUInt32, uint32_t,
                               mediapipe::intops::LogFatalOnError);
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(SafeUInt64, uint64_t,
                               mediapipe::intops::LogFatalOnError);

namespace mediapipe {
namespace intops {

//
// Test cases that apply to signed and unsigned types equally.
//

template <typename T>
class SignNeutralSafeIntTest : public ::testing::Test {
 public:
  typedef T SafeIntTypeUnderTest;
};

typedef ::testing::Types<SafeInt8, SafeUInt8, SafeInt16, SafeUInt16, SafeInt32,
                         SafeUInt32, SafeInt64, SafeUInt64>
    AllSafeIntTypes;

TYPED_TEST_SUITE(SignNeutralSafeIntTest, AllSafeIntTypes);

TYPED_TEST(SignNeutralSafeIntTest, TestCtors) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test default construction.
    T x;
    EXPECT_EQ(V(), x.value());
  }

  {  // Test construction from a value.
    T x(93);
    EXPECT_EQ(V(93), x.value());
  }

  {  // Test copy construction.
    T x(76);
    T y(x);
    EXPECT_EQ(V(76), y.value());
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestUnaryOperators) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test unary plus of positive values.
    T x(123);
    EXPECT_EQ(V(123), (+x).value());
  }
  {  // Test logical not of positive values.
    T x(123);
    EXPECT_EQ(false, !x);
    EXPECT_EQ(true, !!x);
  }
  {  // Test logical not of zero.
    T x(0);
    EXPECT_EQ(true, !x);
    EXPECT_EQ(false, !!x);
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestCtorFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test out-of-bounds construction.
    if (std::numeric_limits<V>::is_signed || sizeof(V) < sizeof(uint64_t)) {
      EXPECT_DEATH((T(std::numeric_limits<uint64_t>::max())), "bounds");
    }
  }
  {  // Test out-of-bounds construction from float.
    EXPECT_DEATH((T(std::numeric_limits<float>::max())), "bounds");
    EXPECT_DEATH((T(-std::numeric_limits<float>::max())), "bounds");
  }
  {  // Test out-of-bounds construction from double.
    EXPECT_DEATH((T(std::numeric_limits<double>::max())), "bounds");
    EXPECT_DEATH((T(-std::numeric_limits<double>::max())), "bounds");
  }
  {  // Test out-of-bounds construction from long double.
    EXPECT_DEATH((T(std::numeric_limits<long double>::max())), "bounds");
    EXPECT_DEATH((T(-std::numeric_limits<long double>::max())), "bounds");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestIncrementDecrement) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test simple increments and decrements.
    T x(0);
    EXPECT_EQ(V(0), x.value());
    EXPECT_EQ(V(0), (x++).value());
    EXPECT_EQ(V(1), x.value());
    EXPECT_EQ(V(2), (++x).value());
    EXPECT_EQ(V(2), x.value());
    EXPECT_EQ(V(2), (x--).value());
    EXPECT_EQ(V(1), x.value());
    EXPECT_EQ(V(0), (--x).value());
    EXPECT_EQ(V(0), x.value());
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestIncrementDecrementFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test overflowing increment.
    T x(std::numeric_limits<V>::max() - 1);
    EXPECT_EQ(std::numeric_limits<V>::max(), (++x).value());
    EXPECT_DEATH(x++, "overflow");
    EXPECT_DEATH(++x, "overflow");
  }
  {  // Test underflowing decrement.
    T x(std::numeric_limits<V>::min() + 1);
    EXPECT_EQ(std::numeric_limits<V>::min(), (--x).value());
    EXPECT_DEATH(x--, "underflow");
    EXPECT_DEATH(--x, "underflow");
  }
}

#define TEST_T_OP_T(xval, op, yval)            \
  {                                            \
    T x(xval);                                 \
    T y(yval);                                 \
    V expected = x.value() op y.value();       \
    EXPECT_EQ(expected, (x op y).value());     \
    EXPECT_EQ(expected, (x op## = y).value()); \
    EXPECT_EQ(expected, x.value());            \
  }

TYPED_TEST(SignNeutralSafeIntTest, TestAdd) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test positive vs. positive addition.
  TEST_T_OP_T(9, +, 3)
  // Test addition by zero.
  TEST_T_OP_T(93, +, 0);
}

TYPED_TEST(SignNeutralSafeIntTest, TestAddFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test overflowing addition.
    T x(std::numeric_limits<V>::max());
    EXPECT_DEATH(x + T(1), "overflow");
    EXPECT_DEATH(x += T(1), "overflow");
  }
  {  // Test overflowing addition.
    T x(std::numeric_limits<V>::max());
    EXPECT_DEATH(x + T(std::numeric_limits<V>::max()), "overflow");
    EXPECT_DEATH(x += T(std::numeric_limits<V>::max()), "overflow");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestSubtract) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test positive vs. positive subtraction.
  TEST_T_OP_T(9, -, 3)
  // Test subtraction of zero.
  TEST_T_OP_T(93, -, 0);
}

TYPED_TEST(SignNeutralSafeIntTest, TestSubtractFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test underflowing subtraction.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x - T(1), "underflow");
    EXPECT_DEATH(x -= T(1), "underflow");
  }
  {  // Test underflowing subtraction.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x - T(std::numeric_limits<V>::max()), "underflow");
    EXPECT_DEATH(x -= T(std::numeric_limits<V>::max()), "underflow");
  }
}

#define TEST_T_OP_NUM(xval, op, numtype, yval) \
  {                                            \
    T x(xval);                                 \
    numtype y = yval;                          \
    V expected = x.value() op y;               \
    EXPECT_EQ(expected, (x op y).value());     \
    EXPECT_EQ(expected, (x op## = y).value()); \
    EXPECT_EQ(expected, x.value());            \
  }

TYPED_TEST(SignNeutralSafeIntTest, TestMultiply) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test positive vs. positive multiplication across types.
  TEST_T_OP_NUM(9, *, int32_t, 3);
  TEST_T_OP_NUM(9, *, uint32_t, 3);
  TEST_T_OP_NUM(9, *, float, 3);
  TEST_T_OP_NUM(9, *, double, 3);

  // Test positive vs. zero multiplication commutatively across types.  This
  // was a real bug.
  TEST_T_OP_NUM(93, *, int32_t, 0);
  TEST_T_OP_NUM(93, *, uint32_t, 0);
  TEST_T_OP_NUM(93, *, float, 0);
  TEST_T_OP_NUM(93, *, double, 0);

  TEST_T_OP_NUM(0, *, int32_t, 76);
  TEST_T_OP_NUM(0, *, uint32_t, 76);
  TEST_T_OP_NUM(0, *, float, 76);
  TEST_T_OP_NUM(0, *, double, 76);

  // Test positive vs. epsilon multiplication.
  TEST_T_OP_NUM(93, *, float, std::numeric_limits<float>::epsilon());
  TEST_T_OP_NUM(93, *, double, std::numeric_limits<float>::epsilon());

  {  // Test multiplication by float.
     // Multiplication is the only operator that takes one numeric type and
     // one StrongInt type *and* is commutative.  This was a real bug.
    T x(0);
    EXPECT_EQ(0, (x * static_cast<float>(1.1)).value());
    EXPECT_EQ(0, (static_cast<float>(1.1) * x).value());
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestMultiplyFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test overflowing multiplication.
    T x(std::numeric_limits<V>::max());
    EXPECT_DEATH(x * 2, "overflow");
    EXPECT_DEATH(x *= 2, "overflow");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestDivide) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test positive vs. positive division across types.
  TEST_T_OP_NUM(9, /, int32_t, 3);
  TEST_T_OP_NUM(9, /, uint32_t, 3);
  TEST_T_OP_NUM(9, /, float, 3);
  TEST_T_OP_NUM(9, /, double, 3);

  // Test zero vs. positive division across types.
  TEST_T_OP_NUM(0, /, int32_t, 76);
  TEST_T_OP_NUM(0, /, uint32_t, 76);
  TEST_T_OP_NUM(0, /, float, 76);
  TEST_T_OP_NUM(0, /, double, 76);
}

TYPED_TEST(SignNeutralSafeIntTest, TestDivideFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test divide by zero.
    T x(93);
    EXPECT_DEATH(x / 0, "divide by zero");
    EXPECT_DEATH(x /= 0, "divide by zero");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestModulo) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test positive vs. positive modulo across signedness.
  TEST_T_OP_NUM(7, %, int32_t, 6);
  TEST_T_OP_NUM(7, %, uint32_t, 6);

  // Test zero vs. positive modulo across signedness.
  TEST_T_OP_NUM(0, %, int32_t, 6);
  TEST_T_OP_NUM(0, %, uint32_t, 6);
}

TYPED_TEST(SignNeutralSafeIntTest, TestModuloFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test modulo by zero.
    T x(93);
    EXPECT_DEATH(x % 0, "divide by zero");
    EXPECT_DEATH(x %= 0, "divide by zero");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestLeftShift) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test basic shift.
  TEST_T_OP_NUM(0x09, <<, int, 3);
  // Test shift by zero.
  TEST_T_OP_NUM(0x09, <<, int, 0);
}

TYPED_TEST(SignNeutralSafeIntTest, TestLeftShiftFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test shift by a negative.
    T x(9);
    EXPECT_DEATH(x << -1, "shift by negative");
    EXPECT_DEATH(x <<= -1, "shift by negative");
  }
  {  // Test shift by a too-large.
    T x(9);
    EXPECT_DEATH(x << sizeof(T) * CHAR_BIT, "shift by large");
    EXPECT_DEATH(x <<= sizeof(T) * CHAR_BIT, "shift by large");
    EXPECT_DEATH(x <<= 0x100000001ULL, "shift by large");
  }
  {  // Test overflowing shift.
    T x(std::numeric_limits<V>::max());
    EXPECT_DEATH(x << 1, "overflow");
    EXPECT_DEATH(x <<= 1, "overflow");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestRightShift) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test basic shift.
  TEST_T_OP_NUM(0x09, >>, int, 3);
  // Test shift by zero.
  TEST_T_OP_NUM(0x09, >>, int, 0);
}

TYPED_TEST(SignNeutralSafeIntTest, TestRightShiftFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test shift by a negative.
    T x(9);
    EXPECT_DEATH(x >> -1, "shift by negative");
    EXPECT_DEATH(x >>= -1, "shift by negative");
  }
  {  // Test shift by a too-large.
    T x(9);
    EXPECT_DEATH(x >> sizeof(T) * CHAR_BIT, "shift by large");
    EXPECT_DEATH(x >>= sizeof(T) * CHAR_BIT, "shift by large");
  }
}

TYPED_TEST(SignNeutralSafeIntTest, TestFloatToIntTruncation) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test construction from float.
  {
    float f = 93.123;
    T x(f);
    EXPECT_EQ(93, x.value());
  }
  {
    float f = 93.76;
    T x(f);
    EXPECT_EQ(93, x.value());
  }
  // Test construction from double.
  {
    double f = 93.123;
    T x(f);
    EXPECT_EQ(93, x.value());
  }
  {
    double f = 93.76;
    T x(f);
    EXPECT_EQ(93, x.value());
  }
  // Test construction from long double.
  {
    long double f = 93.123;
    T x(f);
    EXPECT_EQ(93, x.value());
  }
  {
    long double f = 93.76;
    T x(f);
    EXPECT_EQ(93, x.value());
  }
}

//
// Test cases that apply only to signed types.
//

template <typename T>
class SignedSafeIntTest : public ::testing::Test {
 public:
  typedef T SafeIntTypeUnderTest;
};

typedef ::testing::Types<SafeInt8, SafeInt16, SafeInt32, SafeInt64>
    SignedSafeIntTypes;

TYPED_TEST_SUITE(SignedSafeIntTest, SignedSafeIntTypes);

TYPED_TEST(SignedSafeIntTest, TestCtors) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test construction from a negative value.
    T x(-1);
    EXPECT_EQ(V(-1), x.value());
  }
}

TYPED_TEST(SignedSafeIntTest, TestUnaryOperators) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test unary plus and minus of positive values.
    T x(123);
    EXPECT_EQ(V(123), (+x).value());
    EXPECT_EQ(V(-123), (-x).value());
  }
  {  // Test unary plus and minus of negative values.
    T x(-123);
    EXPECT_EQ(V(-123), (+x).value());
    EXPECT_EQ(V(123), (-x).value());
  }
  {  // Test logical not of negative values.
    T x(-123);
    EXPECT_EQ(false, !x);
    EXPECT_EQ(true, !!x);
  }
}

TYPED_TEST(SignedSafeIntTest, TestUnaryOperatorsFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test unary minus of negative values.
    T y(std::numeric_limits<V>::min());
    EXPECT_DEATH(-y, "overflow");
  }
}

TYPED_TEST(SignedSafeIntTest, TestAdd) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test negative vs. positive addition.
  TEST_T_OP_T(-9, +, 3)
  // Test positive vs. negative addition.
  TEST_T_OP_T(9, +, -3)
  // Test negative vs. negative addition.
  TEST_T_OP_T(-9, +, -3)
}

TYPED_TEST(SignedSafeIntTest, TestAddFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test underflow by addition of a negative.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x + T(-1), "underflow");
    EXPECT_DEATH(x += T(-1), "underflow");
  }
}

TYPED_TEST(SignedSafeIntTest, TestSubtract) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test negative vs. positive subtraction.
  TEST_T_OP_T(-9, -, 3)
  // Test positive vs. negative subtraction.
  TEST_T_OP_T(9, -, -3)
  // Test negative vs. negative subtraction.
  TEST_T_OP_T(-9, -, -3)
  // Test positive vs. positive subtraction resulting in negative.
  TEST_T_OP_T(3, -, 9);
  // Test subtraction from zero.
  TEST_T_OP_T(0, -, 93);
}

TYPED_TEST(SignedSafeIntTest, TestSubtractFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test overflow by subtraction of a negative.
    T x(std::numeric_limits<V>::max());
    EXPECT_DEATH(x - T(-1), "overflow");
    EXPECT_DEATH(x -= T(-1), "overflow");
  }
}

TYPED_TEST(SignedSafeIntTest, TestMultiply) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test negative vs. positive multiplication across types.
  TEST_T_OP_NUM(-9, *, int32_t, 3);
  TEST_T_OP_NUM(-9, *, uint32_t, 3);
  TEST_T_OP_NUM(-9, *, float, 3);
  TEST_T_OP_NUM(-9, *, double, 3);
  // Test positive vs. negative multiplication across types.
  TEST_T_OP_NUM(9, *, int32_t, -3);
  // Don't cover unsigneds that are initialized from negative values.
  TEST_T_OP_NUM(9, *, float, -3);
  TEST_T_OP_NUM(9, *, double, -3);
  // Test negative vs. negative multiplication across types.
  TEST_T_OP_NUM(-9, *, int32_t, -3);
  // Don't cover unsigneds that are initialized from negative values.
  TEST_T_OP_NUM(-9, *, float, -3);
  TEST_T_OP_NUM(-9, *, double, -3);

  // Test negative vs. zero multiplication commutatively across types.
  TEST_T_OP_NUM(-93, *, int32_t, 0);
  TEST_T_OP_NUM(-93, *, uint32_t, 0);
  TEST_T_OP_NUM(-93, *, float, 0);
  TEST_T_OP_NUM(-93, *, double, 0);
  TEST_T_OP_NUM(0, *, int32_t, -76);
  TEST_T_OP_NUM(0, *, uint32_t, -76);
  TEST_T_OP_NUM(0, *, float, -76);
  TEST_T_OP_NUM(0, *, double, -76);

  // Test negative vs. epsilon multiplication.
  TEST_T_OP_NUM(-93, *, float, std::numeric_limits<float>::epsilon());
  TEST_T_OP_NUM(-93, *, double, std::numeric_limits<float>::epsilon());
}

TYPED_TEST(SignedSafeIntTest, TestMultiplyFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test underflowing multiplication.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x * 2, "underflow");
    EXPECT_DEATH(x *= 2, "underflow");
  }
  {  // Test underflowing multiplication.
    T x(std::numeric_limits<V>::max());
    EXPECT_DEATH(x * -2, "underflow");
    EXPECT_DEATH(x *= -2, "underflow");
  }
  {  // Test overflowing multiplication.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x * -2, "overflow");
    EXPECT_DEATH(x *= -2, "overflow");
  }
  {  // Test overflowing multiplication.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x * -1, "overflow");
    EXPECT_DEATH(x *= -1, "overflow");
  }
  {  // Test underflowing multiplication where rhs type is uint64_t.
    T x(-2);
    EXPECT_DEATH(x * std::numeric_limits<uint64_t>::max(), "underflow");
    EXPECT_DEATH(x *= std::numeric_limits<uint64_t>::max(), "underflow");
  }
}

TYPED_TEST(SignedSafeIntTest, TestDivide) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test negative vs. positive division across types.
  TEST_T_OP_NUM(-9, /, int32_t, 3);
  TEST_T_OP_NUM(-9, /, uint32_t, 3);
  TEST_T_OP_NUM(-9, /, float, 3);
  TEST_T_OP_NUM(-9, /, double, 3);
  // Test positive vs. negative division across types.
  TEST_T_OP_NUM(9, /, int32_t, -3);
  TEST_T_OP_NUM(9, /, uint32_t, -3);
  TEST_T_OP_NUM(9, /, float, -3);
  TEST_T_OP_NUM(9, /, double, -3);
  // Test negative vs. negative division across types.
  TEST_T_OP_NUM(-9, /, int32_t, -3);
  TEST_T_OP_NUM(-9, /, uint32_t, -3);
  TEST_T_OP_NUM(-9, /, float, -3);
  TEST_T_OP_NUM(-9, /, double, -3);

  // Test zero vs. negative division across types.
  TEST_T_OP_NUM(0, /, int32_t, -76);
  TEST_T_OP_NUM(0, /, uint32_t, -76);
  TEST_T_OP_NUM(0, /, float, -76);
  TEST_T_OP_NUM(0, /, double, -76);
}

TYPED_TEST(SignedSafeIntTest, TestDivideFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test overflowing division.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x / -1, "overflow");
    EXPECT_DEATH(x /= -1, "overflow");
  }
}

TYPED_TEST(SignedSafeIntTest, TestModulo) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  // Test negative vs. positive modulo across signedness.
  TEST_T_OP_NUM(-7, %, int32_t, 6);
  TEST_T_OP_NUM(-7, %, uint32_t, 6);
  // Test positive vs. negative modulo across signedness.
  TEST_T_OP_NUM(7, %, int32_t, -6);
  TEST_T_OP_NUM(7, %, uint32_t, -6);
  // Test negative vs. negative modulo across signedness.
  TEST_T_OP_NUM(-7, %, int32_t, -6);
  TEST_T_OP_NUM(-7, %, uint32_t, -6);

  // Test zero vs. negative modulo across signedness.
  TEST_T_OP_NUM(0, %, int32_t, -6);
  TEST_T_OP_NUM(0, %, uint32_t, -6);
}

TYPED_TEST(SignedSafeIntTest, TestModuloFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test overflowing modulo.
    T x(std::numeric_limits<V>::min());
    EXPECT_DEATH(x % -1, "overflow");
    EXPECT_DEATH(x %= -1, "overflow");
  }
}

TYPED_TEST(SignedSafeIntTest, TestLeftShiftFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test shift of a negative.
    T x(-9);
    EXPECT_DEATH(x << 1, "shift of negative");
    EXPECT_DEATH(x <<= 1, "shift of negative");
  }
}

TYPED_TEST(SignedSafeIntTest, TestRightShiftFailures) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test shift of a negative.
    T x(-9);
    EXPECT_DEATH(x >> 1, "shift of negative");
    EXPECT_DEATH(x >>= 1, "shift of negative");
  }
}

//
// Test cases that apply only to unsigned types.
//

template <typename T>
class UnsignedSafeIntTest : public ::testing::Test {
 public:
  typedef T SafeIntTypeUnderTest;
};

typedef ::testing::Types<SafeUInt8, SafeUInt16, SafeUInt32, SafeUInt64>
    UnsignedSafeIntTypes;

TYPED_TEST_SUITE(UnsignedSafeIntTest, UnsignedSafeIntTypes);

TYPED_TEST(UnsignedSafeIntTest, TestCtors) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test out-of-bounds construction.
    EXPECT_DEATH(T(-1), "bounds");
  }
  {  // Test out-of-bounds construction from float.
    EXPECT_DEATH((T(static_cast<float>(-1))), "bounds");
  }
  {  // Test out-of-bounds construction from double.
    EXPECT_DEATH((T(static_cast<double>(-1))), "bounds");
  }
  {  // Test out-of-bounds construction from long double.
    EXPECT_DEATH((T(static_cast<long double>(-1))), "bounds");
  }
}

TYPED_TEST(UnsignedSafeIntTest, TestUnaryOperators) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test bitwise not of positive values.
    T x(123);
    EXPECT_EQ(V(~(x.value())), (~x).value());
    EXPECT_EQ(x.value(), (~~x).value());
  }
  {  // Test bitwise not of zero.
    T x(0x00);
    EXPECT_EQ(V(~(x.value())), (~x).value());
    EXPECT_EQ(x.value(), (~~x).value());
  }
}

TYPED_TEST(UnsignedSafeIntTest, TestMultiply) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test multiplication by a negative.
    T x(93);
    EXPECT_DEATH(x * -1, "negation");
    EXPECT_DEATH(x *= -1, "negation");
  }
}

TYPED_TEST(UnsignedSafeIntTest, TestDivide) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test division by a negative.
    T x(93);
    EXPECT_DEATH(x / -1, "negation");
    EXPECT_DEATH(x /= -1, "negation");
  }
}

TYPED_TEST(UnsignedSafeIntTest, TestModulo) {
  typedef typename TestFixture::SafeIntTypeUnderTest T;
  typedef typename T::ValueType V;

  {  // Test modulo by a negative.
    T x(93);
    EXPECT_DEATH(x % -5, "negation");
    EXPECT_DEATH(x %= -5, "negation");
  }
}

}  // namespace intops
}  // namespace mediapipe
