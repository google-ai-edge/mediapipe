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

// Test functions in MathUtil.

#include "mediapipe/framework/deps/mathutil.h"

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>

#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace {

TEST(MathUtil, Round) {
  // test float rounding
  EXPECT_EQ(mediapipe::MathUtil::FastIntRound(0.7f), 1);
  EXPECT_EQ(mediapipe::MathUtil::FastIntRound(5.7f), 6);
  EXPECT_EQ(mediapipe::MathUtil::FastIntRound(6.3f), 6);
  EXPECT_EQ(mediapipe::MathUtil::FastIntRound(1000000.7f), 1000001);

  // test that largest representable number below 0.5 rounds to zero.
  // this is important because naive implementation of round:
  // static_cast<int>(r + 0.5f) is 1 due to implicit rounding in operator+
  float rf = std::nextafter(0.5f, .0f);
  EXPECT_LT(rf, 0.5f);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(rf), 0);

  // same test for double
  double rd = std::nextafter(0.5, 0.0);
  EXPECT_LT(rd, 0.5);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(rd), 0);

  // same test for long double
  long double rl = std::nextafter(0.5l, 0.0l);
  EXPECT_LT(rl, 0.5l);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(rl), 0);
}

static void BM_IntCast(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += static_cast<int>(x);
    x += 0.1;
    sum += static_cast<int>(x);
    x += 0.1;
    sum += static_cast<int>(x);
    x += 0.1;
    sum += static_cast<int>(x);
    x += 0.1;
    sum += static_cast<int>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_IntCast);

static void BM_Int64Cast(benchmark::State& state) {
  double x = 0.1;
  int64_t sum = 0;
  for (auto _ : state) {
    sum += static_cast<int64_t>(x);
    x += 0.1;
    sum += static_cast<int64_t>(x);
    x += 0.1;
    sum += static_cast<int64_t>(x);
    x += 0.1;
    sum += static_cast<int64_t>(x);
    x += 0.1;
    sum += static_cast<int64_t>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_Int64Cast);

static void BM_IntRound(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::Round<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_IntRound);

static void BM_FastIntRound(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::FastIntRound(x);
    x += 0.1;
    sum += mediapipe::MathUtil::FastIntRound(x);
    x += 0.1;
    sum += mediapipe::MathUtil::FastIntRound(x);
    x += 0.1;
    sum += mediapipe::MathUtil::FastIntRound(x);
    x += 0.1;
    sum += mediapipe::MathUtil::FastIntRound(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_FastIntRound);

static void BM_Int64Round(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::Round<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<int64_t>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_Int64Round);

static void BM_UintRound(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::Round<uint32_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<uint32_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<uint32_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<uint32_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::Round<uint32_t>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_UintRound);

static void BM_SafeIntCast(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::SafeCast<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_SafeIntCast);

static void BM_SafeInt64Cast(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::SafeCast<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeCast<int64_t>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_SafeInt64Cast);

static void BM_SafeIntRound(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::SafeRound<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_SafeIntRound);

static void BM_SafeInt64Round(benchmark::State& state) {
  double x = 0.1;
  int sum = 0;
  for (auto _ : state) {
    sum += mediapipe::MathUtil::SafeRound<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int64_t>(x);
    x += 0.1;
    sum += mediapipe::MathUtil::SafeRound<int64_t>(x);
    x += 0.1;
  }
  EXPECT_NE(sum, 0);  // Don't let 'sum' get optimized away.
}
BENCHMARK(BM_SafeInt64Round);

TEST(MathUtil, IntRound) {
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(0.0), 0);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(0.49), 0);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(1.49), 1);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(-0.49), 0);
  EXPECT_EQ(mediapipe::MathUtil::Round<int>(-1.49), -1);

  // Either adjacent integer is an acceptable result.
  EXPECT_EQ(fabs(mediapipe::MathUtil::Round<int>(0.5) - 0.5), 0.5);
  EXPECT_EQ(fabs(mediapipe::MathUtil::Round<int>(1.5) - 1.5), 0.5);
  EXPECT_EQ(fabs(mediapipe::MathUtil::Round<int>(-0.5) + 0.5), 0.5);
  EXPECT_EQ(fabs(mediapipe::MathUtil::Round<int>(-1.5) + 1.5), 0.5);

  EXPECT_EQ(mediapipe::MathUtil::Round<int>(static_cast<double>(0x76543210)),
            0x76543210);

  // A double-precision number has a 53-bit mantissa (52 fraction bits),
  // so the following value can be represented exactly.
  int64_t value64 = static_cast<int64_t>(0x1234567890abcd00);
  EXPECT_EQ(mediapipe::MathUtil::Round<int64_t>(static_cast<double>(value64)),
            value64);
}

template <class F>
F NextAfter(F x, F y);

template <>
float NextAfter(float x, float y) {
  return nextafterf(x, y);
}

template <>
double NextAfter(double x, double y) {
  return nextafter(x, y);
}

template <class FloatIn, class IntOut>
class SafeCastTester {
 public:
  static void Run() {
    const IntOut imax = std::numeric_limits<IntOut>::max();
    EXPECT_GT(imax, 0);
    const IntOut imin = std::numeric_limits<IntOut>::min();
    const bool s = std::numeric_limits<IntOut>::is_signed;
    if (s) {
      EXPECT_LT(imin, 0);
    } else {
      EXPECT_EQ(0, imin);
    }

    // Some basic tests.
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(0.0)),
              0);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-0.0)),
              0);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(0.99)),
              0);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(1.0)),
              1);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(1.01)),
              1);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(1.99)),
              1);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(2.0)),
              2);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(2.01)),
              2);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-0.99)), 0);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-1.0)),
              s ? -1 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-1.01)),
        s ? -1 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-1.99)),
        s ? -1 : 0);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-2.0)),
              s ? -2 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-2.01)),
        s ? -2 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(117.9)),
        117);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(118.0)),
        118);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(118.1)),
        118);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-117.9)),
        s ? -117 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-118.0)),
        s ? -118 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(-118.1)),
        s ? -118 : 0);

    // Some edge cases.
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                  std::numeric_limits<FloatIn>::max()),
              imax);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                  -std::numeric_limits<FloatIn>::max()),
              imin);
    const FloatIn inf_val = std::numeric_limits<FloatIn>::infinity();
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(inf_val), imax);
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(-inf_val), imin);
    const FloatIn nan_val = inf_val - inf_val;
    EXPECT_TRUE(std::isnan(nan_val));
    EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(nan_val), 0);

    // Some larger numbers.
    if (sizeof(IntOut) >= 32) {
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(0x76543210)),
                0x76543210);
    }

    if (sizeof(FloatIn) >= 64) {
      // A double-precision number has a 53-bit mantissa (52 fraction bits),
      // so the following value can be represented exactly by a double.
      int64_t value64 = static_cast<int64_t>(0x1234567890abcd00);
      const IntOut expected =
          (sizeof(IntOut) >= 64) ? static_cast<IntOut>(value64) : imax;
      EXPECT_EQ(
          mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(value64)),
          expected);
    }

    // Check values near imin and imax
    static const int kLoopCount = 10;

    {
      // Values greater than or equal to imax should convert to imax
      FloatIn v = static_cast<FloatIn>(imax);
      for (int i = 0; i < kLoopCount; i++) {
        EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(v), imax);
        EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                      static_cast<FloatIn>(v + 10000.)),
                  imax);
        v = NextAfter(v, std::numeric_limits<FloatIn>::max());
      }
    }

    {
      // Values less than or equal to imin should convert to imin
      FloatIn v = static_cast<FloatIn>(imin);
      for (int i = 0; i < kLoopCount; i++) {
        EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(v), imin);
        EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                      static_cast<FloatIn>(v - 10000.)),
                  imin);
        v = NextAfter(v, -std::numeric_limits<FloatIn>::max());
      }
    }

    {
      // Values slightly less than imax which can be exactly represented as a
      // FloatIn should convert exactly to themselves.
      IntOut v = imax;
      for (int i = 0; i < kLoopCount; i++) {
        v = std::min<IntOut>(v - 1,
                             NextAfter(static_cast<FloatIn>(v),
                                       -std::numeric_limits<FloatIn>::max()));
        EXPECT_EQ(
            mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(v)), v);
      }
    }

    {
      // Values slightly greater than imin which can be exactly represented as a
      // FloatIn should convert exactly to themselves.
      IntOut v = imin;
      for (int i = 0; i < kLoopCount; i++) {
        v = std::max<IntOut>(v + 1,
                             NextAfter(static_cast<FloatIn>(v),
                                       std::numeric_limits<FloatIn>::max()));
        EXPECT_EQ(
            mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(v)), v);
      }
    }

    // When FloatIn is wider than IntOut, we can test that fractional conversion
    // near imax works as expected.
    if (sizeof(FloatIn) > sizeof(IntOut)) {
      {
        // Values slightly less than imax should convert to imax - 1
        FloatIn v = static_cast<FloatIn>(imax);
        for (int i = 0; i < kLoopCount; i++) {
          v = NextAfter(static_cast<FloatIn>(v),
                        -std::numeric_limits<FloatIn>::max());
          EXPECT_EQ(
              mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(v)),
              imax - 1);
        }
      }
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.1)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.99)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 1.0)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 1.99)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 2.0)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.1)),
                imax - 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.99)),
                imax - 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.0)),
                imax - 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.01)),
                imax - 2);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.99)),
                imax - 2);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 2.0)),
                imax - 2);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 2.01)),
                imax - 3);
    }
    // When FloatIn is wider than IntOut, and IntOut is signed, we can test
    // that fractional conversion near imin works as expected.
    if (s && (sizeof(FloatIn) > sizeof(IntOut))) {
      {
        // Values just over imin should convert to imin + 1
        FloatIn v = static_cast<FloatIn>(imin);
        for (int i = 0; i < kLoopCount; i++) {
          v = NextAfter(static_cast<FloatIn>(v),
                        std::numeric_limits<FloatIn>::max());
          EXPECT_EQ(
              mediapipe::MathUtil::SafeCast<IntOut>(static_cast<FloatIn>(v)),
              imin + 1);
        }
      }
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.1)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.99)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 1.0)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.99)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 2.0)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 0.1)),
                imin + 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 0.99)),
                imin + 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 1.0)),
                imin + 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 1.01)),
                imin + 2);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 1.99)),
                imin + 2);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 2.0)),
                imin + 2);
      EXPECT_EQ(mediapipe::MathUtil::SafeCast<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 2.01)),
                imin + 3);
    }
  }
};

TEST(MathUtil, SafeCast) {
  SafeCastTester<float, int8_t>::Run();
  SafeCastTester<double, int8_t>::Run();
  SafeCastTester<float, int16_t>::Run();
  SafeCastTester<double, int16_t>::Run();
  SafeCastTester<float, int32_t>::Run();
  SafeCastTester<double, int32_t>::Run();
  SafeCastTester<float, int64_t>::Run();
  SafeCastTester<double, int64_t>::Run();
  SafeCastTester<float, uint8_t>::Run();
  SafeCastTester<double, uint8_t>::Run();
  SafeCastTester<float, uint16_t>::Run();
  SafeCastTester<double, uint16_t>::Run();
  SafeCastTester<float, uint32_t>::Run();
  SafeCastTester<double, uint32_t>::Run();
  SafeCastTester<float, uint64_t>::Run();
  SafeCastTester<double, uint64_t>::Run();

  // Spot-check SafeCast<int>
  EXPECT_EQ(mediapipe::MathUtil::SafeCast<int>(static_cast<float>(12345.678)),
            12345);
  EXPECT_EQ(mediapipe::MathUtil::SafeCast<int>(static_cast<float>(12345.4321)),
            12345);
  EXPECT_EQ(mediapipe::MathUtil::SafeCast<int>(static_cast<double>(-12345.678)),
            -12345);
  EXPECT_EQ(
      mediapipe::MathUtil::SafeCast<int>(static_cast<double>(-12345.4321)),
      -12345);
  EXPECT_EQ(mediapipe::MathUtil::SafeCast<int>(1E47), 2147483647);
  EXPECT_EQ(mediapipe::MathUtil::SafeCast<int>(-1E47),
            static_cast<int64_t>(-2147483648));
}

template <class FloatIn, class IntOut>
class SafeRoundTester {
 public:
  static void Run() {
    const IntOut imax = std::numeric_limits<IntOut>::max();
    EXPECT_GT(imax, 0);
    const IntOut imin = std::numeric_limits<IntOut>::min();
    const bool s = std::numeric_limits<IntOut>::is_signed;
    if (s) {
      EXPECT_LT(imin, 0);
    } else {
      EXPECT_EQ(0, imin);
    }

    // Some basic tests.
    EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(0.0)),
              0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-0.0)), 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(0.49)), 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(0.51)), 1);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(1.49)), 1);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(1.51)), 2);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-0.49)), 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-0.51)),
        s ? -1 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-1.49)),
        s ? -1 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-1.51)),
        s ? -2 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(117.4)),
        117);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(117.6)),
        118);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-117.4)),
        s ? -117 : 0);
    EXPECT_EQ(
        mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-117.6)),
        s ? -118 : 0);

    // At the midpoint between ints, either adjacent int is an acceptable
    // result.
    EXPECT_EQ(
        fabs(mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(0.5)) -
             0.5),
        0.5);
    EXPECT_EQ(
        fabs(mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(1.5)) -
             1.5),
        0.5);
    EXPECT_EQ(fabs(mediapipe::MathUtil::SafeRound<IntOut>(
                       static_cast<FloatIn>(117.5)) -
                   117.5),
              0.5);
    if (s) {
      EXPECT_EQ(fabs(mediapipe::MathUtil::SafeRound<IntOut>(
                         static_cast<FloatIn>(-0.5)) +
                     0.5),
                0.5);
      EXPECT_EQ(fabs(mediapipe::MathUtil::SafeRound<IntOut>(
                         static_cast<FloatIn>(-1.5)) +
                     1.5),
                0.5);
      EXPECT_EQ(fabs(mediapipe::MathUtil::SafeRound<IntOut>(
                         static_cast<FloatIn>(-117.5)) +
                     117.5),
                0.5);
    } else {
      EXPECT_EQ(
          mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-0.5)),
          0);
      EXPECT_EQ(
          mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-1.5)),
          0);
      EXPECT_EQ(
          mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(-117.5)),
          0);
    }

    // Some edge cases.
    EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                  std::numeric_limits<FloatIn>::max()),
              imax);
    EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                  -std::numeric_limits<FloatIn>::max()),
              imin);
    const FloatIn inf_val = std::numeric_limits<FloatIn>::infinity();
    EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(inf_val), imax);
    EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(-inf_val), imin);
    const FloatIn nan_val = inf_val - inf_val;
    EXPECT_TRUE(std::isnan(nan_val));
    EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(nan_val), 0);

    // Some larger numbers.
    if (sizeof(IntOut) >= 32) {
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(0x76543210)),
                0x76543210);
    }

    if (sizeof(FloatIn) >= 64) {
      // A double-precision number has a 53-bit mantissa (52 fraction bits),
      // so the following value can be represented exactly by a double.
      int64_t value64 = static_cast<int64_t>(0x1234567890abcd00);
      const IntOut expected =
          (sizeof(IntOut) >= 64) ? static_cast<IntOut>(value64) : imax;
      EXPECT_EQ(
          mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(value64)),
          expected);
    }

    // Check values near imin and imax
    static const int kLoopCount = 10;

    {
      // Values greater than or equal to imax should round to imax
      FloatIn v = static_cast<FloatIn>(imax);
      for (int i = 0; i < kLoopCount; i++) {
        EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(v), imax);
        EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                      static_cast<FloatIn>(v + 10000.)),
                  imax);
        v = NextAfter(v, std::numeric_limits<FloatIn>::max());
      }
    }

    {
      // Values less than or equal to imin should round to imin
      FloatIn v = static_cast<FloatIn>(imin);
      for (int i = 0; i < kLoopCount; i++) {
        EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(v), imin);
        EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                      static_cast<FloatIn>(v - 10000.)),
                  imin);
        v = NextAfter(v, -std::numeric_limits<FloatIn>::max());
      }
    }

    {
      // Values slightly less than imax which can be exactly represented as a
      // FloatIn should round exactly to themselves.
      IntOut v = imax;
      for (int i = 0; i < kLoopCount; i++) {
        v = std::min<IntOut>(v - 1,
                             NextAfter(static_cast<FloatIn>(v),
                                       -std::numeric_limits<FloatIn>::max()));
        EXPECT_EQ(
            mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(v)), v);
      }
    }

    {
      // Values slightly greater than imin which can be exactly represented as a
      // FloatIn should round exactly to themselves.
      IntOut v = imin;
      for (int i = 0; i < kLoopCount; i++) {
        v = std::max<IntOut>(v + 1,
                             NextAfter(static_cast<FloatIn>(v),
                                       std::numeric_limits<FloatIn>::max()));
        EXPECT_EQ(
            mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(v)), v);
      }
    }

    // When FloatIn is wider than IntOut, we can test that fractional rounding
    // near imax works as expected.
    if (sizeof(FloatIn) > sizeof(IntOut)) {
      {
        // Values slightly less than imax should round to imax
        FloatIn v = static_cast<FloatIn>(imax);
        for (int i = 0; i < kLoopCount; i++) {
          v = NextAfter(static_cast<FloatIn>(v),
                        -std::numeric_limits<FloatIn>::max());
          EXPECT_EQ(
              mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(v)),
              imax);
        }
      }
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.1)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.49)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.5)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.51)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.99)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.1)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.49)),
                imax);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.51)),
                imax - 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.99)),
                imax - 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.49)),
                imax - 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.51)),
                imax - 2);
    }
    // When FloatIn is wider than IntOut, or if IntOut is unsigned, we can test
    // that fractional rounding near imin works as expected.
    if (!s || (sizeof(FloatIn) > sizeof(IntOut))) {
      {
        // Values slightly greater than imin should round to imin
        FloatIn v = static_cast<FloatIn>(imin);
        for (int i = 0; i < kLoopCount; i++) {
          v = NextAfter(static_cast<FloatIn>(v),
                        std::numeric_limits<FloatIn>::max());
          EXPECT_EQ(
              mediapipe::MathUtil::SafeRound<IntOut>(static_cast<FloatIn>(v)),
              imin);
        }
      }
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.1)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.49)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.5)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.51)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) - 0.99)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 0.1)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 0.49)),
                imin);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 0.51)),
                imin + 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 0.99)),
                imin + 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 1.49)),
                imin + 1);
      EXPECT_EQ(mediapipe::MathUtil::SafeRound<IntOut>(
                    static_cast<FloatIn>(static_cast<FloatIn>(imin) + 1.51)),
                imin + 2);
    }
  }
};

TEST(MathUtil, SafeRound) {
  SafeRoundTester<float, int8_t>::Run();
  SafeRoundTester<double, int8_t>::Run();
  SafeRoundTester<float, int16_t>::Run();
  SafeRoundTester<double, int16_t>::Run();
  SafeRoundTester<float, int32_t>::Run();
  SafeRoundTester<double, int32_t>::Run();
  SafeRoundTester<float, int64_t>::Run();
  SafeRoundTester<double, int64_t>::Run();
  SafeRoundTester<float, uint8_t>::Run();
  SafeRoundTester<double, uint8_t>::Run();
  SafeRoundTester<float, uint16_t>::Run();
  SafeRoundTester<double, uint16_t>::Run();
  SafeRoundTester<float, uint32_t>::Run();
  SafeRoundTester<double, uint32_t>::Run();
  SafeRoundTester<float, uint64_t>::Run();
  SafeRoundTester<double, uint64_t>::Run();

  // Spot-check SafeRound<int>
  EXPECT_EQ(mediapipe::MathUtil::SafeRound<int>(static_cast<float>(12345.678)),
            12346);
  EXPECT_EQ(mediapipe::MathUtil::SafeRound<int>(static_cast<float>(12345.4321)),
            12345);
  EXPECT_EQ(
      mediapipe::MathUtil::SafeRound<int>(static_cast<double>(-12345.678)),
      -12346);
  EXPECT_EQ(
      mediapipe::MathUtil::SafeRound<int>(static_cast<double>(-12345.4321)),
      -12345);
  EXPECT_EQ(mediapipe::MathUtil::SafeRound<int>(1E47), 2147483647);
  EXPECT_EQ(mediapipe::MathUtil::SafeRound<int>(-1E47),
            static_cast<int64_t>(-2147483648));
}

}  // namespace
