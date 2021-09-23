// Copyright 2020 The MediaPipe Authors.
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

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_test.pb.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

namespace test_ns {

constexpr char kOutTag[] = "OUT";
constexpr char kInTag[] = "IN";

class TestSinkCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kInTag).Set<mediapipe::InputOnlyProto>();
    cc->Outputs().Tag(kOutTag).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    int x = cc->Inputs().Tag(kInTag).Get<mediapipe::InputOnlyProto>().x();
    cc->Outputs().Tag(kOutTag).AddPacket(
        MakePacket<int>(x).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TestSinkCalculator);

}  // namespace test_ns

TEST(PacketTest, InputTypeRegistration) {
  using testing::Contains;
  ASSERT_EQ(mediapipe::InputOnlyProto{}.GetTypeName(),
            "mediapipe.InputOnlyProto");
  EXPECT_THAT(packet_internal::MessageHolderRegistry::GetRegisteredNames(),
              Contains("mediapipe.InputOnlyProto"));
}

}  // namespace
}  // namespace mediapipe
