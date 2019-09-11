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

#include "mediapipe/framework/packet_generator.h"

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

namespace {
class DoNothingGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    for (CollectionItemId id = input_side_packets->BeginId();
         id < input_side_packets->EndId(); ++id) {
      input_side_packets->Get(id).SetAny();
    }
    for (CollectionItemId id = output_side_packets->BeginId();
         id < output_side_packets->EndId(); ++id) {
      output_side_packets->Get(id).Set<bool>();
    }
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& extendable_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    for (CollectionItemId id = output_side_packets->BeginId();
         id < output_side_packets->EndId(); ++id) {
      output_side_packets->Get(id) = MakePacket<bool>(true);
    }
    return ::mediapipe::OkStatus();
  }
};

REGISTER_PACKET_GENERATOR(DoNothingGenerator);

TEST(PacketGeneratorTest, FillExpectationsOnConfig) {
  PacketGeneratorConfig config;
  config.set_packet_generator("DoNothingGenerator");
  config.add_input_side_packet("any");
  config.add_input_side_packet("number");
  config.add_input_side_packet("of_inputs");
  config.add_output_side_packet("any_number_of");
  config.add_output_side_packet("output_side_packets");
  MP_EXPECT_OK(tool::RunGeneratorFillExpectations(config));
}

}  // namespace
}  // namespace mediapipe
