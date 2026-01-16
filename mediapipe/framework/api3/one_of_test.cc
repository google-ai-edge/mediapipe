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
#include "mediapipe/framework/api3/one_of.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/functional/overload.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {
using ::testing::Pointee;

struct ToStringNode : Node<"ToStringNode"> {
  template <typename S>
  struct Contract {
    Input<S, OneOf<int, float>> in{"IN"};
    Output<S, std::string> str{"STR"};
  };
};

class ToStringNodeImpl : public Calculator<ToStringNode, ToStringNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<ToStringNode>& cc) final {
    RET_CHECK(cc.in);
    std::string str;
    if (cc.in.Has<int>()) {
      str = absl::StrFormat("%d", cc.in.GetOrDie<int>());
    } else if (cc.in.Has<float>()) {
      str = absl::StrFormat("%.3ff", cc.in.GetOrDie<float>());
    }
    cc.str.Send(std::move(str));
    return absl::OkStatus();
  };
};

template <typename T>
Stream<std::string> ToString(GenericGraph& graph, Stream<T> in) {
  auto& node = graph.AddNode<ToStringNode>();
  node.in.Set(in);
  return node.str.Get();
}

TEST(OneOfTest, CanUseOneOfForCalculatorInputs) {
  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner, Runner::For(ToString<int>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<int>(42)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "42");
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner, Runner::For(ToString<float>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<float>(0.5f)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "0.500f");
  }
}

struct ToStringWithVisitNode : Node<"ToStringWithVisitNode"> {
  template <typename S>
  struct Contract {
    Input<S, OneOf<int, float, double>> in{"IN"};
    Output<S, std::string> str{"STR"};
  };
};

class ToStringWithVisitNodeImpl
    : public Calculator<ToStringWithVisitNode, ToStringWithVisitNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<ToStringWithVisitNode>& cc) final {
    RET_CHECK(cc.in);
    cc.str.Send(cc.in.VisitOrDie(
        [](int value) { return absl::StrFormat("%d", value); },
        [](float value) { return absl::StrFormat("%.3ff", value); },
        [](double value) { return absl::StrFormat("%.3f", value); }));
    return absl::OkStatus();
  };
};

template <typename T>
Stream<std::string> ToStringWithVisit(GenericGraph& graph, Stream<T> in) {
  auto& node = graph.AddNode<ToStringWithVisitNode>();
  node.in.Set(in);
  return node.str.Get();
}

TEST(OneOfTest, CanUseOneOfWithVsitiForCalculatorInputs) {
  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner,
                            Runner::For(ToStringWithVisit<int>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<int>(42)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "42");
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner,
                            Runner::For(ToStringWithVisit<float>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<float>(0.5f)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "0.500f");
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner,
                            Runner::For(ToStringWithVisit<double>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<double>(0.001)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "0.001");
  }
}

struct ToStringWithPacketOrDieNode : Node<"ToStringWithPacketOrDieNode"> {
  template <typename S>
  struct Contract {
    Input<S, OneOf<int, float>> in{"IN"};
    Output<S, std::string> str{"STR"};
  };
};

class ToStringWithPacketOrDieImpl
    : public Calculator<ToStringWithPacketOrDieNode,
                        ToStringWithPacketOrDieImpl> {
 public:
  absl::Status Process(
      CalculatorContext<ToStringWithPacketOrDieNode>& cc) final {
    if (cc.in.Has<int>()) {
      Packet<int> p = cc.in.PacketOrDie<int>();
      RET_CHECK(p);
      cc.str.Send(absl::StrFormat("%d", p.GetOrDie()));
      return absl::OkStatus();
    }
    if (cc.in.Has<float>()) {
      Packet<float> p = cc.in.PacketOrDie<float>();
      RET_CHECK(p);
      cc.str.Send(absl::StrFormat("%.3ff", p.GetOrDie()));
      return absl::OkStatus();
    }
    return absl::InternalError("Input is missing.");
  };
};

template <typename T>
Stream<std::string> ToStringWithPacketOrDie(GenericGraph& graph, Stream<T> in) {
  auto& node = graph.AddNode<ToStringWithPacketOrDieNode>();
  node.in.Set(in);
  return node.str.Get();
}

TEST(OneOfTest, CanUseOneOfWithPacketOrDieForCalculatorInputs) {
  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner,
                            Runner::For(ToStringWithPacketOrDie<int>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<int>(42)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "42");
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto runner, Runner::For(ToStringWithPacketOrDie<float>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<float>(0.5f)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "0.500f");
  }
}

struct ToStringWithSingleVisitorNode : Node<"ToStringWithSingleVisitorNode"> {
  template <typename S>
  struct Contract {
    Input<S, OneOf<int, float, double>> in{"IN"};
    Output<S, std::string> str{"STR"};
  };
};

class ToStringWithSingleVisitorNodeImpl
    : public Calculator<ToStringWithSingleVisitorNode,
                        ToStringWithSingleVisitorNodeImpl> {
 public:
  absl::Status Process(
      CalculatorContext<ToStringWithSingleVisitorNode>& cc) final {
    RET_CHECK(cc.in);
    // We pass a single visitor covering all variants.
    cc.str.Send(
        cc.in.VisitOrDie([](auto value) { return absl::StrCat(value); }));
    return absl::OkStatus();
  };
};

template <typename T>
Stream<std::string> ToStringWithSingleVisitor(GenericGraph& graph,
                                              Stream<T> in) {
  auto& node = graph.AddNode<ToStringWithSingleVisitorNode>();
  node.in.Set(in);
  return node.str.Get();
}

TEST(OneOfTest, CanUseOneOfWithSingleVisitorForCalculatorInputs) {
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto runner, Runner::For(ToStringWithSingleVisitor<int>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<int>(42)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "42");
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto runner, Runner::For(ToStringWithSingleVisitor<float>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<float>(0.5f)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "0.5");
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto runner, Runner::For(ToStringWithSingleVisitor<double>).Create());
    MP_ASSERT_OK_AND_ASSIGN(Packet<std::string> str,
                            runner.Run(MakePacket<double>(0.001)));

    ASSERT_TRUE(str);
    EXPECT_EQ(str.GetOrDie(), "0.001");
  }
}

struct DemuxWithVisitAsPacketNode : Node<"DemuxWithVisitAsPacketNode"> {
  template <typename S>
  struct Contract {
    // We use `std::unique_ptr` to make sure the implementation does not copy
    // the content but is efficiently using the underlying shared packet.
    Input<S, OneOf<std::unique_ptr<int>, std::unique_ptr<float>>> in{"IN"};
    Output<S, std::unique_ptr<int>> ints{"INTS"};
    Output<S, std::unique_ptr<float>> floats{"FLOATS"};
  };
};

class DemuxWithVisitAsPacketNodeImpl
    : public Calculator<DemuxWithVisitAsPacketNode,
                        DemuxWithVisitAsPacketNodeImpl> {
 public:
  absl::Status Process(
      CalculatorContext<DemuxWithVisitAsPacketNode>& cc) final {
    RET_CHECK(cc.in);
    cc.in.VisitAsPacketOrDie(absl::Overload(
        [&](Packet<std::unique_ptr<int>> packet) {
          cc.ints.Send(std::move(packet));
        },
        [&](Packet<std::unique_ptr<float>> packet) {
          cc.floats.Send(std::move(packet));
        }));
    return absl::OkStatus();
  };
};

template <typename InputType>
std::tuple<Stream<std::unique_ptr<int>>, Stream<std::unique_ptr<float>>>
DemuxWithVisitAsPacket(GenericGraph& graph,
                       Stream<std::unique_ptr<InputType>> in) {
  auto& node = graph.AddNode<DemuxWithVisitAsPacketNode>();
  node.in.Set(in);
  return {node.ints.Get(), node.floats.Get()};
}

TEST(OneOfTest, VisitAsPacket) {
  {
    MP_ASSERT_OK_AND_ASSIGN(auto runner,
                            Runner::For(DemuxWithVisitAsPacket<int>).Create());
    MP_ASSERT_OK_AND_ASSIGN((auto [ints, floats]),
                            runner.Run(MakePacket<std::unique_ptr<int>>(
                                std::make_unique<int>(42))));
    ASSERT_TRUE(ints);
    EXPECT_THAT(ints.GetOrDie(), Pointee(42));
    EXPECT_FALSE(floats);
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto runner, Runner::For(DemuxWithVisitAsPacket<float>).Create());
    MP_ASSERT_OK_AND_ASSIGN((auto [ints, floats]),
                            runner.Run(MakePacket<std::unique_ptr<float>>(
                                std::make_unique<float>(0.5f))));
    ASSERT_TRUE(floats);
    EXPECT_THAT(floats.GetOrDie(), Pointee(0.5f));
    EXPECT_FALSE(ints);
  }
}

}  // namespace
}  // namespace mediapipe::api3
