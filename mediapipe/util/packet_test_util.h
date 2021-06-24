// Copyright 2021 The MediaPipe Authors.
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
// Utilities that help to make assertions about packet contents in tests.

#ifndef MEDIAPIPE_UTIL_PACKET_TEST_UTIL_H_
#define MEDIAPIPE_UTIL_PACKET_TEST_UTIL_H_

#include <ostream>
#include <string>
#include <typeinfo>

#include "mediapipe/framework/demangle.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

namespace internal {

template <typename PayloadType>
class PacketMatcher : public ::testing::MatcherInterface<const Packet&> {
 public:
  template <typename InnerMatcher>
  explicit PacketMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(
            ::testing::SafeMatcherCast<const PayloadType&>(inner_matcher)) {}

  // Returns true iff the packet contains value of PayloadType satisfying
  // the inner matcher.
  bool MatchAndExplain(
      const Packet& packet,
      ::testing::MatchResultListener* listener) const override {
    if (!packet.ValidateAsType<PayloadType>().ok()) {
      *listener << packet.DebugString() << " does not contain expected type "
                << ExpectedTypeName();
      return false;
    }
    ::testing::StringMatchResultListener match_listener;
    const PayloadType& payload = packet.Get<PayloadType>();
    const bool matches =
        inner_matcher_.MatchAndExplain(payload, &match_listener);
    const std::string explanation = match_listener.str();
    *listener << packet.DebugString() << " containing value "
              << ::testing::PrintToString(payload);
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
    return ::mediapipe::Demangle(typeid(PayloadType).name());
  }

  const ::testing::Matcher<const PayloadType&> inner_matcher_;
};

}  // namespace internal

// Creates matcher validating that the packet contains value of expected type
// and satisfying the provided inner matcher.
//
// PayloadType template parameter has to be specified explicitly, but matcher
// type can be inferred. Example:
//
// EXPECT_THAT(MakePacket<int>(42), PacketContains<int>(Eq(42)))
template <typename PayloadType, typename InnerMatcher>
inline ::testing::Matcher<const Packet&> PacketContains(
    InnerMatcher inner_matcher) {
  return ::testing::MakeMatcher(
      new internal::PacketMatcher<PayloadType>(inner_matcher));
}

// Creates matcher validating the packet's timestamp satisfies the provided
// timestamp_matcher. It also checks that the packet contains value of expected
// type and satisfies the provided content matcher.
//
// PayloadType template parameter has to be specified explicitly, but matcher
// type can be inferred. Example:
//
// EXPECT_THAT(MakePacket<int>(42).At(Timestamp(20)),
//             PacketContainsTimestampAndPayload<int>(  //
//                Eq(Timestamp(20)),
//                Eq(42)))
template <typename PayloadType, typename TimestampMatcher,
          typename ContentMatcher>
inline ::testing::Matcher<const Packet&> PacketContainsTimestampAndPayload(
    TimestampMatcher timestamp_matcher, ContentMatcher content_matcher) {
  return testing::AllOf(
      testing::Property("Packet::Timestamp", &Packet::Timestamp,
                        timestamp_matcher),
      PacketContains<PayloadType>(content_matcher));
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_PACKET_TEST_UTIL_H_
