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

#include "mediapipe/framework/test_service.h"

namespace mediapipe {

const GraphService<TestServiceObject> kTestService("test_service");
const GraphService<int> kAnotherService("another_service");

::mediapipe::Status TestServiceCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<int>();
  cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
  // This service will be required. The graph won't start without it.
  cc->UseService(kTestService);
  // This service is optional for this calculator.
  cc->UseService(kAnotherService).Optional();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TestServiceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  // For an optional service, check whether it's available.
  if (cc->Service(kAnotherService).IsAvailable()) {
    optional_bias_ = cc->Service(kAnotherService).GetObject();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TestServiceCalculator::Process(CalculatorContext* cc) {
  int value = cc->Inputs().Index(0).Value().Get<int>();
  // A required service is sure to be available, so we can just GetObject.
  TestServiceObject& service_object = cc->Service(kTestService).GetObject();
  int delta = service_object["delta"];
  service_object["count"] += 1;
  int x = value + delta + optional_bias_;
  cc->Outputs().Index(0).Add(new int(x), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(TestServiceCalculator);

}  // namespace mediapipe
