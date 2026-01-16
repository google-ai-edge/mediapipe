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

#import <Foundation/Foundation.h>

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"

#include "mediapipe/framework/packet.h"

/**
 * This class helps create various kinds of packets for MediaPipe Audio Tasks.
 */
@interface MPPAudioPacketCreator : NSObject

/**
 * Creates a MediapPipe Packet wrapping the buffer of a `MPPAudioData` that can be send to a graph.
 *
 * @param audioData The audio data of type `MPPAudioData` to send to the MediaPipe graph.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 *    error will be saved.
 *
 * @return The MediaPipe packet containing the buffer of the given audio data. An empty packet is
 * returned if an error occurred during the conversion.
 */
+ (mediapipe::Packet)createPacketWithAudioData:(MPPAudioData *)audioData error:(NSError **)error;

/**
 * Creates a MediapPipe Packet wrapping the buffer of a `MPPAudioData` that can be send to a graph
 * at the specified timestamp.
 *
 * @param audioData The audio data of type `MPPAudioData` to send to the MediaPipe graph.
 * @param timestampInMilliseconds The timestamp (in milliseconds) to assign to the packet.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 *    error will be saved.
 *
 * @return The MediaPipe packet containing the buffer of the given audio data. An empty packet is
 * returned if an error occurred during the conversion.
 */
+ (mediapipe::Packet)createPacketWithAudioData:(MPPAudioData *)audioData
                       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                         error:(NSError **)error;

@end
