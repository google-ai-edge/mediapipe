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

#include "mediapipe/framework/timestamp.h"

#include <memory>
#include <vector>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

TEST(TimestampDeathTest, ConstructorDieOnSpecialValue) {
  EXPECT_DEATH(Timestamp value1(Timestamp::Unset().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::Unstarted().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::PreStream().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::Min().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::Max().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::PostStream().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::OneOverPostStream().Value()), "");
  EXPECT_DEATH(Timestamp value1(Timestamp::Done().Value()), "");
}

TEST(TimestampDeathTest, Overflow) {
  Timestamp large = Timestamp(kint64max / 2 + 100);
  TimestampDiff large_diff = TimestampDiff(kint64max / 2 + 100);
  Timestamp small = Timestamp(kint64min / 2 - 100);
  EXPECT_FALSE(large.IsSpecialValue());
  EXPECT_FALSE(small.IsSpecialValue());
  EXPECT_DEATH(large_diff + large_diff, "");
  EXPECT_DEATH(-large_diff - large_diff, "");
  EXPECT_DEATH(small - large, "");
  EXPECT_EQ(TimestampDiff(0), large - large);
  EXPECT_EQ(TimestampDiff(0), small - small);
  EXPECT_DEATH(Timestamp::PostStream() + 0, "");
  // Test out-of-bounds construction from seconds. int64max is roughly
  // 9.2e18 < 1.0e19. So 1.0e13 seconds = 1.0e19 microseconds is out of
  // bounds.
  EXPECT_DEATH(Timestamp::FromSeconds(1.0e13), "bounds");
}

TEST(TimestampTest, Constructor) { Timestamp value1(1); }

TEST(TimestampTest, IsSpecial) {
  Timestamp unset1;
  Timestamp unset2 = Timestamp::Unset();
  Timestamp unstarted = Timestamp::Unstarted();
  Timestamp pre_stream = Timestamp::PreStream();
  Timestamp beginning = Timestamp::Min();
  Timestamp smallest_normal = Timestamp::Min() + 1;
  Timestamp zero = Timestamp(0);
  Timestamp largest_normal = Timestamp::Max() - 1;
  Timestamp limit = Timestamp::Max();
  Timestamp post_stream = Timestamp::PostStream();
  Timestamp one_over_post_stream = Timestamp::OneOverPostStream();
  Timestamp done = Timestamp::Done();

  EXPECT_EQ(unset1, unset2);

  EXPECT_TRUE(unset1.IsSpecialValue());
  EXPECT_TRUE(unset2.IsSpecialValue());
  EXPECT_TRUE(unstarted.IsSpecialValue());
  EXPECT_TRUE(pre_stream.IsSpecialValue());
  EXPECT_TRUE(beginning.IsSpecialValue());
  EXPECT_FALSE(smallest_normal.IsSpecialValue());
  EXPECT_FALSE(zero.IsSpecialValue());
  EXPECT_FALSE(largest_normal.IsSpecialValue());
  EXPECT_TRUE(limit.IsSpecialValue());
  EXPECT_TRUE(post_stream.IsSpecialValue());
  EXPECT_TRUE(one_over_post_stream.IsSpecialValue());
  EXPECT_TRUE(done.IsSpecialValue());

  EXPECT_FALSE(unset1.IsRangeValue());
  EXPECT_FALSE(unset2.IsRangeValue());
  EXPECT_FALSE(unstarted.IsRangeValue());
  EXPECT_FALSE(pre_stream.IsRangeValue());
  EXPECT_TRUE(beginning.IsRangeValue());
  EXPECT_TRUE(smallest_normal.IsRangeValue());
  EXPECT_TRUE(zero.IsRangeValue());
  EXPECT_TRUE(largest_normal.IsRangeValue());
  EXPECT_TRUE(limit.IsRangeValue());
  EXPECT_FALSE(post_stream.IsRangeValue());
  EXPECT_FALSE(one_over_post_stream.IsRangeValue());
  EXPECT_FALSE(done.IsRangeValue());

  EXPECT_FALSE(unset1.IsAllowedInStream());
  EXPECT_FALSE(unset2.IsAllowedInStream());
  EXPECT_FALSE(unstarted.IsAllowedInStream());
  EXPECT_TRUE(pre_stream.IsAllowedInStream());
  EXPECT_TRUE(beginning.IsAllowedInStream());
  EXPECT_TRUE(smallest_normal.IsAllowedInStream());
  EXPECT_TRUE(zero.IsAllowedInStream());
  EXPECT_TRUE(largest_normal.IsAllowedInStream());
  EXPECT_TRUE(limit.IsAllowedInStream());
  EXPECT_TRUE(post_stream.IsAllowedInStream());
  EXPECT_FALSE(one_over_post_stream.IsAllowedInStream());
  EXPECT_FALSE(done.IsAllowedInStream());
}

TEST(TimestampTest, NextAllowedInStream) {
  EXPECT_EQ(Timestamp::OneOverPostStream(),
            Timestamp::PreStream().NextAllowedInStream());
  EXPECT_EQ(Timestamp::Min() + 1, Timestamp::Min().NextAllowedInStream());
  EXPECT_EQ(Timestamp::Min() + 2, (Timestamp::Min() + 1).NextAllowedInStream());
  EXPECT_EQ(Timestamp(-999), Timestamp(-1000).NextAllowedInStream());
  EXPECT_EQ(Timestamp(1), Timestamp(0).NextAllowedInStream());
  EXPECT_EQ(Timestamp(1001), Timestamp(1000).NextAllowedInStream());
  EXPECT_EQ(Timestamp::Max() - 1, (Timestamp::Max() - 2).NextAllowedInStream());
  EXPECT_EQ(Timestamp::Max(), (Timestamp::Max() - 1).NextAllowedInStream());
  EXPECT_EQ(Timestamp::OneOverPostStream(),
            Timestamp::Max().NextAllowedInStream());
  EXPECT_EQ(Timestamp::OneOverPostStream(),
            Timestamp::PostStream().NextAllowedInStream());
}

TEST(TimestampTest, SpecialValueDifferences) {
  {  // Lower range
    const std::vector<Timestamp> timestamps = {
        Timestamp::Unset(), Timestamp::Unstarted(), Timestamp::PreStream(),
        Timestamp::Min()};
    for (int i = 1; i < timestamps.size(); ++i) {
      EXPECT_EQ(1, timestamps[i].Value() - timestamps[i - 1].Value());
    }
  }
  {  // Upper range
    const std::vector<Timestamp> timestamps = {
        Timestamp::Max(), Timestamp::PostStream(),
        Timestamp::OneOverPostStream(), Timestamp::Done()};
    for (int i = 1; i < timestamps.size(); ++i) {
      EXPECT_EQ(1, timestamps[i].Value() - timestamps[i - 1].Value());
    }
  }
}

TEST(TimestampTest, Differences) {
  Timestamp t0 = Timestamp(0);
  Timestamp t10 = Timestamp(10);
  Timestamp t20 = Timestamp(20);

  TimestampDiff d0(0);
  TimestampDiff d10(10);
  TimestampDiff d20(20);
  TimestampDiff dn10(-10);

  TimestampDiff d0_1 = t0 - t0;
  TimestampDiff d0_2 = t10 - t10;
  TimestampDiff d10_1 = t20 - t10;
  TimestampDiff d10_2 = t10 - t0;
  TimestampDiff dn10_1 = t0 - t10;
  TimestampDiff dn10_2 = t10 - t20;

  EXPECT_EQ(d0, d0_1);
  EXPECT_EQ(d0, d0_2);
  EXPECT_EQ(d10, d10_1);
  EXPECT_EQ(d10, d10_2);
  EXPECT_EQ(dn10, dn10_1);
  EXPECT_EQ(dn10, dn10_2);

  EXPECT_GT(t10, t0);
  EXPECT_GT(t20, t10);
  EXPECT_GE(t0, t0);
  EXPECT_GE(t10, t0);
  EXPECT_GE(t10, t10);
  EXPECT_GE(t20, t10);
  EXPECT_GE(t20, t20);

  EXPECT_LT(t0, t10);
  EXPECT_LT(t10, t20);
  EXPECT_LE(t0, t0);
  EXPECT_LE(t0, t10);
  EXPECT_LE(t10, t10);
  EXPECT_LE(t10, t20);
  EXPECT_LE(t20, t20);

  EXPECT_FALSE(t10 > t10);
  EXPECT_FALSE(t10 < t10);

  EXPECT_EQ(d10, d0 + d10);
  EXPECT_EQ(d10, d10 + d0);
  EXPECT_EQ(d10, d20 - d10);
  EXPECT_EQ(d20, d10 + d10);
  EXPECT_EQ(d0, d10 - d10);

  EXPECT_EQ(t10, t0 + d10);
  EXPECT_EQ(t10, d10 + t0);
  EXPECT_EQ(t0, t10 - d10);
  EXPECT_EQ(t0, -d10 + t10);

  EXPECT_GT(d10, d0);
  EXPECT_GT(d20, d10);
  EXPECT_GE(d0, d0);
  EXPECT_GE(d10, d0);
  EXPECT_GE(d10, d10);
  EXPECT_GE(d20, d10);
  EXPECT_GE(d20, d20);

  EXPECT_LT(d0, d10);
  EXPECT_LT(d10, d20);
  EXPECT_LE(d0, d0);
  EXPECT_LE(d0, d10);
  EXPECT_LE(d10, d10);
  EXPECT_LE(d10, d20);
  EXPECT_LE(d20, d20);

  EXPECT_FALSE(d10 > d10);
  EXPECT_FALSE(d10 < d10);
}

TEST(TimestampTest, Clamping) {
  EXPECT_EQ(Timestamp::Max(), (Timestamp::Max() - 100) + 100);
  EXPECT_EQ(Timestamp::Max(), (Timestamp::Max() - 100) + 200);
  EXPECT_EQ(Timestamp::Max() - 1, (Timestamp::Max() - 100) + 99);

  EXPECT_EQ(Timestamp::Min(), (Timestamp::Min() + 100) - 100);
  EXPECT_EQ(Timestamp::Min(), (Timestamp::Min() + 100) - 200);
  EXPECT_EQ(Timestamp::Min() + 1, (Timestamp::Min() + 100) - 99);

  EXPECT_NE(Timestamp::Max(), Timestamp::Max() - 100);
  EXPECT_NE(Timestamp::Min(), Timestamp::Min() + 100);
}

TEST(TimestampTest, IncrementInPlace) {
  Timestamp val(100);
  val += 100;
  EXPECT_EQ(Timestamp(200), val);
  val += TimestampDiff(1);
  EXPECT_EQ(Timestamp(201), val);
  val -= TimestampDiff(51);
  EXPECT_EQ(Timestamp(150), val);
  val -= 150;
  EXPECT_EQ(Timestamp(0), val);
  EXPECT_EQ(Timestamp(10), val += 10);
  EXPECT_EQ(Timestamp(-40), val -= 50);
  EXPECT_EQ(Timestamp(-40), val);

  EXPECT_EQ(Timestamp(-40), val++);
  EXPECT_EQ(Timestamp(-39), val);
  EXPECT_EQ(Timestamp(-38), ++val);
  EXPECT_EQ(Timestamp(-38), val);

  EXPECT_EQ(Timestamp(-38), val--);
  EXPECT_EQ(Timestamp(-39), val);
  EXPECT_EQ(Timestamp(-40), --val);
  EXPECT_EQ(Timestamp(-40), val);
}

TEST(TimestampTest, AddZeroToMinAndMax) {
  EXPECT_EQ(Timestamp::Max(), Timestamp::Max() + TimestampDiff(0));
  EXPECT_EQ(Timestamp::Min(), Timestamp::Min() + TimestampDiff(0));
  EXPECT_EQ(Timestamp::Max(), Timestamp::Max() - TimestampDiff(0));
  EXPECT_EQ(Timestamp::Min(), Timestamp::Min() - TimestampDiff(0));
}

// Note: Add test to timestamp_pcoder_test.cc if another special value is added.
TEST(TimestampTest, DebugString) {
  EXPECT_EQ(std::string("Timestamp::Unset()"),
            Timestamp::Unset().DebugString());
  EXPECT_EQ(std::string("Timestamp::Unstarted()"),
            Timestamp::Unstarted().DebugString());
  EXPECT_EQ(std::string("Timestamp::PreStream()"),
            Timestamp::PreStream().DebugString());
  EXPECT_EQ(std::string("Timestamp::Min()"), Timestamp::Min().DebugString());
  EXPECT_EQ(std::string("-100"), Timestamp(-100).DebugString());
  EXPECT_EQ(std::string("0"), Timestamp(0).DebugString());
  EXPECT_EQ(std::string("100"), Timestamp(100).DebugString());
  EXPECT_EQ(std::string("Timestamp::Max()"), Timestamp::Max().DebugString());
  EXPECT_EQ(std::string("Timestamp::PostStream()"),
            Timestamp::PostStream().DebugString());
  EXPECT_EQ(std::string("Timestamp::OneOverPostStream()"),
            Timestamp::OneOverPostStream().DebugString());
  EXPECT_EQ(std::string("Timestamp::Done()"), Timestamp::Done().DebugString());
}

}  // namespace
}  // namespace mediapipe
