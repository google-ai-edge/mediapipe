// Copyright 2022 The MediaPipe Authors.
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

#import <Foundation/Foundation.h>

#include "google/protobuf/any.pb.h"
#include "mediapipe/framework/calculator_options.pb.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Any MediaPipe task options should confirm to this protocol.
 */
@protocol MPPTaskOptionsProtocol <NSObject>

@optional

/**
 * Copies the iOS MediaPipe task options to an object of `mediapipe::CalculatorOptions` proto.
 */
- (void)copyToProto:(::mediapipe::CalculatorOptions *)optionsProto;

/**
 * Copies the iOS MediaPipe task options to an object of `google::protobuf::Any` proto.
 */
- (void)copyToAnyProto:(::google::protobuf::Any *)optionsProto;

@end

NS_ASSUME_NONNULL_END
