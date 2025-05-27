// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_API3_PACKET_MATCHER_H_
#define MEDIAPIPE_FRAMEWORK_API3_PACKET_MATCHER_H_

#include <ostream>
#include <string>

#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/demangle.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3 {

template <typename PayloadT>
class PacketMatcher
    : public testing::MatcherInterface<const Packet<PayloadT>&> {
 public:
  template <typename InnerMatcher>
  explicit PacketMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(
            testing::SafeMatcherCast<const PayloadT&>(inner_matcher)) {}

  // Returns true if the packet contains value of PayloadT satisfying the
  // inner matcher.
  bool MatchAndExplain(const Packet<PayloadT>& packet,
                       testing::MatchResultListener* listener) const override {
    if (!packet) {
      *listener << packet.DebugString() << " is empty";
      return false;
    }

    testing::StringMatchResultListener match_listener;
    const PayloadT& payload = packet.GetOrDie();
    const bool matches =
        inner_matcher_.MatchAndExplain(payload, &match_listener);
    const std::string& explanation = match_listener.str();
    *listener << packet.DebugString() << " containing value "
              << testing::PrintToString(payload);
    if (!explanation.empty()) {
      *listener << ", which " << explanation;
    }
    return matches;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "packet contains value of type " << ExpectedTypeName() << " that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "packet does not contain value of type " << ExpectedTypeName()
        << " that ";
    inner_matcher_.DescribeNegationTo(os);
  }

 private:
  static std::string ExpectedTypeName() {
    return ::mediapipe::Demangle(typeid(PayloadT).name());
  }

  const testing::Matcher<const PayloadT&> inner_matcher_;
};

// EXPECT_THAT(MakePacket<int>(42).At(Timestamp(20)),
//             PacketEq<int>(Eq(42), Eq(Timestamp(20))));
template <typename PayloadT, typename TimestampMatcher, typename ContentMatcher>
inline testing::Matcher<const Packet<PayloadT>&> PacketEq(
    ContentMatcher content_matcher, TimestampMatcher timestamp_matcher) {
  return testing::AllOf(
      testing::MakeMatcher(new PacketMatcher<PayloadT>(content_matcher)),
      testing::Property("Packet::Timestamp", &Packet<PayloadT>::Timestamp,
                        timestamp_matcher));
}

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_PACKET_MATCHER_H_
