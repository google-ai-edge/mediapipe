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

#import <Foundation/Foundation.h>

#include "mediapipe/framework/packet.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

/**
 * This class helps create various kinds of packets for Mediapipe Vision Tasks.
 */
@interface MPPVisionPacketCreator : NSObject

+ (mediapipe::Packet)createPacketWithMPPImage:(MPPImage *)image error:(NSError **)error;

@end
