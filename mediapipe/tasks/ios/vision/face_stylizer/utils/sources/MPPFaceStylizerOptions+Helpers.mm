// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/vision/face_stylizer/utils/sources/MPPFaceStylizerOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/face_stylizer/proto/face_stylizer_graph_options.pb.h"

namespace {
using CalculatorOptionsProto = mediapipe::CalculatorOptions;
using FaceStylizerGraphOptionsProto =
    ::mediapipe::tasks::vision::face_stylizer::proto::FaceStylizerGraphOptions;
}  // namespace

@implementation MPPFaceStylizerOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  FaceStylizerGraphOptionsProto *faceStylizerGraphOptionsProto =
      optionsProto->MutableExtension(FaceStylizerGraphOptionsProto::ext);
  faceStylizerGraphOptionsProto->Clear();

  [self.baseOptions copyToProto:faceStylizerGraphOptionsProto->mutable_base_options()];
}

@end
