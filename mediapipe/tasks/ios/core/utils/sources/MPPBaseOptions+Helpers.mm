// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

namespace {
using BaseOptionsProto = ::mediapipe::tasks::core::proto::BaseOptions;
using CppBaseOptions = ::mediapipe::tasks::core::BaseOptions;
}  // namespace

@implementation MPPBaseOptions (Helpers)

- (void)copyToProto:(BaseOptionsProto *)baseOptionsProto withUseStreamMode:(BOOL)useStreamMode {
  [self copyToProto:baseOptionsProto withUseStreamMode:useStreamMode withUseLitert:NO];
}

- (void)copyToProto:(BaseOptionsProto *)baseOptionsProto
    withUseStreamMode:(BOOL)useStreamMode
        withUseLitert:(BOOL)useLitert {
  [self copyToProto:baseOptionsProto withUseLitert:useLitert];
  baseOptionsProto->set_use_stream_mode(useStreamMode);
}

- (void)copyToProto:(BaseOptionsProto *)baseOptionsProto {
  [self copyToProto:baseOptionsProto withUseLitert:NO];
}

- (void)copyToProto:(BaseOptionsProto *)baseOptionsProto withUseLitert:(BOOL)useLitert {
  CppBaseOptions cppBaseOptions;
  if (self.modelAssetPath) {
    cppBaseOptions.model_asset_path = self.modelAssetPath.UTF8String;
  }
  switch (self.delegate) {
    case MPPDelegateGPU:
      cppBaseOptions.delegate = CppBaseOptions::Delegate::GPU;
      break;
    case MPPDelegateCPU:
    default:
      cppBaseOptions.delegate = CppBaseOptions::Delegate::CPU;
      break;
  }

  *baseOptionsProto =
      ::mediapipe::tasks::core::ConvertBaseOptionsToProto(&cppBaseOptions, useLitert);
}

@end
