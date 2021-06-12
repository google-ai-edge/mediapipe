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

#ifndef MEDIAPIPE_DEPS_MESSAGE_MATCHERS_H_
#define MEDIAPIPE_DEPS_MESSAGE_MATCHERS_H_

#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"

namespace mediapipe {

namespace internal {
bool EqualsMessage(const proto_ns::MessageLite& m_1,
                   const proto_ns::MessageLite& m_2) {
  std::string s_1, s_2;
  m_1.SerializeToString(&s_1);
  m_2.SerializeToString(&s_2);
  return s_1 == s_2;
}
}  // namespace internal

template <typename MessageType>
class ProtoMatcher : public testing::MatcherInterface<MessageType> {
  using MatchResultListener = testing::MatchResultListener;

 public:
  explicit ProtoMatcher(const MessageType& message) : message_(message) {}
  virtual bool MatchAndExplain(MessageType m, MatchResultListener*) const {
    return internal::EqualsMessage(message_, m);
  }

  virtual void DescribeTo(::std::ostream* os) const {
#if defined(MEDIAPIPE_PROTO_LITE)
    *os << "Protobuf messages have identical serializations.";
#else
    *os << message_.DebugString();
#endif
  }

 private:
  const MessageType message_;
};

template <typename MessageType>
inline testing::PolymorphicMatcher<ProtoMatcher<MessageType>> EqualsProto(
    const MessageType& message) {
  return testing::PolymorphicMatcher<ProtoMatcher<MessageType>>(
      ProtoMatcher<MessageType>(message));
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_MESSAGE_MATCHERS_H_
