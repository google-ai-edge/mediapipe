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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/modules/face_geometry/env_generator_calculator.pb.h"
#include "mediapipe/modules/face_geometry/libs/validation_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"

namespace mediapipe {
namespace {

static constexpr char kEnvironmentTag[] = "ENVIRONMENT";

// A calculator that generates an environment, which describes a virtual scene.
//
// Output side packets:
//   ENVIRONMENT (`face_geometry::Environment`, required)
//     Describes an environment; includes the camera frame origin point location
//     as well as virtual camera parameters.
//
// Options:
//   environment (`face_geometry.Environment`, required):
//     Defines an environment to be packed as the output side packet.
//
//     Must be valid (for details, please refer to the proto message definition
//     comments and/or `modules/face_geometry/libs/validation_utils.h/cc`)
//
class EnvGeneratorCalculator : public CalculatorBase {
 public:
  static mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets()
        .Tag(kEnvironmentTag)
        .Set<face_geometry::Environment>();
    return mediapipe::OkStatus();
  }

  mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));

    const face_geometry::Environment& environment =
        cc->Options<FaceGeometryEnvGeneratorCalculatorOptions>().environment();

    MP_RETURN_IF_ERROR(face_geometry::ValidateEnvironment(environment))
        << "Invalid environment!";

    cc->OutputSidePackets()
        .Tag(kEnvironmentTag)
        .Set(mediapipe::MakePacket<face_geometry::Environment>(environment));

    return mediapipe::OkStatus();
  }

  mediapipe::Status Process(CalculatorContext* cc) override {
    return mediapipe::OkStatus();
  }

  mediapipe::Status Close(CalculatorContext* cc) override {
    return mediapipe::OkStatus();
  }
};

}  // namespace

using FaceGeometryEnvGeneratorCalculator = EnvGeneratorCalculator;

REGISTER_CALCULATOR(FaceGeometryEnvGeneratorCalculator);

}  // namespace mediapipe
