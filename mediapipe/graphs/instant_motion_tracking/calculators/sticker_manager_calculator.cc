// Copyright 2020 Google LLC
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

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/instant_motion_tracking/calculators/sticker_buffer.pb.h"
#include "mediapipe/graphs/instant_motion_tracking/calculators/transformations.h"

namespace mediapipe {

constexpr char kProtoDataString[] = "PROTO";
constexpr char kAnchorsTag[] = "ANCHORS";
constexpr char kUserRotationsTag[] = "USER_ROTATIONS";
constexpr char kUserScalingsTag[] = "USER_SCALINGS";
constexpr char kRenderDescriptorsTag[] = "RENDER_DATA";

// This calculator takes in the sticker protobuffer data and parses each
// individual sticker object into anchors, user rotations and scalings, in
// addition to basic render data represented in integer form.
//
// Input:
//  PROTO - String of sticker data in appropriate protobuf format [REQUIRED]
// Output:
//  ANCHORS - Anchors with initial normalized X,Y coordinates [REQUIRED]
//  USER_ROTATIONS - UserRotations with radians of rotation from user [REQUIRED]
//  USER_SCALINGS - UserScalings with increment of scaling from user [REQUIRED]
//  RENDER_DATA - Descriptors of which objects/animations to render for stickers
//  [REQUIRED]
//
// Example config:
// node {
//   calculator: "StickerManagerCalculator"
//   input_stream: "PROTO:sticker_proto_string"
//   output_stream: "ANCHORS:initial_anchor_data"
//   output_stream: "USER_ROTATIONS:user_rotation_data"
//   output_stream: "USER_SCALINGS:user_scaling_data"
//   output_stream: "RENDER_DATA:sticker_render_data"
// }

class StickerManagerCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kProtoDataString));
    RET_CHECK(cc->Outputs().HasTag(kAnchorsTag) &&
              cc->Outputs().HasTag(kUserRotationsTag) &&
              cc->Outputs().HasTag(kUserScalingsTag) &&
              cc->Outputs().HasTag(kRenderDescriptorsTag));

    cc->Inputs().Tag(kProtoDataString).Set<std::string>();
    cc->Outputs().Tag(kAnchorsTag).Set<std::vector<Anchor>>();
    cc->Outputs().Tag(kUserRotationsTag).Set<std::vector<UserRotation>>();
    cc->Outputs().Tag(kUserScalingsTag).Set<std::vector<UserScaling>>();
    cc->Outputs().Tag(kRenderDescriptorsTag).Set<std::vector<int>>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    std::string sticker_proto_string =
        cc->Inputs().Tag(kProtoDataString).Get<std::string>();

    std::vector<Anchor> initial_anchor_data;
    std::vector<UserRotation> user_rotation_data;
    std::vector<UserScaling> user_scaling_data;
    std::vector<int> render_data;

    ::mediapipe::StickerRoll sticker_roll;
    bool parse_success = sticker_roll.ParseFromString(sticker_proto_string);

    // Ensure parsing was a success
    RET_CHECK(parse_success) << "Error parsing sticker protobuf data";

    for (int i = 0; i < sticker_roll.sticker().size(); ++i) {
      // Declare empty structures for sticker data
      Anchor initial_anchor;
      UserRotation user_rotation;
      UserScaling user_scaling;
      // Get individual Sticker object as defined by Protobuffer
      ::mediapipe::Sticker sticker = sticker_roll.sticker(i);
      // Set individual data structure ids to associate with this sticker
      initial_anchor.sticker_id = sticker.id();
      user_rotation.sticker_id = sticker.id();
      user_scaling.sticker_id = sticker.id();
      initial_anchor.x = sticker.x();
      initial_anchor.y = sticker.y();
      initial_anchor.z = 1.0f;  // default to 1.0 in normalized 3d space
      user_rotation.rotation_radians = sticker.rotation();
      user_scaling.scale_factor = sticker.scale();
      const int render_id = sticker.render_id();
      // Set all vector data with sticker attributes
      initial_anchor_data.emplace_back(initial_anchor);
      user_rotation_data.emplace_back(user_rotation);
      user_scaling_data.emplace_back(user_scaling);
      render_data.emplace_back(render_id);
    }

    if (cc->Outputs().HasTag(kAnchorsTag)) {
      cc->Outputs()
          .Tag(kAnchorsTag)
          .AddPacket(MakePacket<std::vector<Anchor>>(initial_anchor_data)
                         .At(cc->InputTimestamp()));
    }
    if (cc->Outputs().HasTag(kUserRotationsTag)) {
      cc->Outputs()
          .Tag(kUserRotationsTag)
          .AddPacket(MakePacket<std::vector<UserRotation>>(user_rotation_data)
                         .At(cc->InputTimestamp()));
    }
    if (cc->Outputs().HasTag(kUserScalingsTag)) {
      cc->Outputs()
          .Tag(kUserScalingsTag)
          .AddPacket(MakePacket<std::vector<UserScaling>>(user_scaling_data)
                         .At(cc->InputTimestamp()));
    }
    if (cc->Outputs().HasTag(kRenderDescriptorsTag)) {
      cc->Outputs()
          .Tag(kRenderDescriptorsTag)
          .AddPacket(MakePacket<std::vector<int>>(render_data)
                         .At(cc->InputTimestamp()));
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Close(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(StickerManagerCalculator);
}  // namespace mediapipe
